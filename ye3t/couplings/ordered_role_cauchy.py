"""Factorized Young/O(3) coordinates for ordered role-resolved Cauchy inputs.

Each factor is a full role-by-magnetic multiplet.  Local diagonal Young
couplings are exact metric-dual maps; global Young induction and binary CG
remain separate.  The full ordered orbit-by-coordinate matrix is not formed.
"""

from functools import lru_cache
from itertools import product
import hashlib
import json
import math

from .young_coset_transport import (
    adjacent_generator_columns, bind_young_restriction,
    canonical_factor_coset, direct_restricted_young_values,
    expanded_young_values, transported_young_slice,
)


ORDERED_ROLE_COPY_GAUGE = "full_family_cholesky_orthonormal_v1"


def _local_choices(request, block_index):
    from ye3t.core.basis.exhaustive_enumeration import integer_partitions
    from ye3t.representations.kronecker_intertwiners import kronecker_multiplicity

    from . import lifted_cauchy_scalar as scalar

    size = int(request["block_sizes"][block_index])
    ell = int(request["channels"][block_index]["l"])
    role_dimension = int(request["role_dimension"])
    choices = []
    partitions = tuple(tuple(int(value) for value in item)
                       for item in integer_partitions(size))
    for parent in partitions:
        for role in partitions:
            if len(role) > role_dimension or (
                request["kappa_policy"] == "trivial" and role != (size,)
            ):
                continue
            role_copies = int(scalar._hook_content_dimension(role, role_dimension))
            if not role_copies:
                continue
            for angular in partitions:
                kronecker_copies = int(kronecker_multiplicity(
                    parent, role, angular
                ))
                if not kronecker_copies:
                    continue
                for Lambda, angular_copies in sorted(
                    scalar._angular_counts(size, ell, angular).items()
                ):
                    for role_copy, angular_copy, kronecker_copy in product(
                        range(role_copies), range(int(angular_copies)),
                        range(kronecker_copies),
                    ):
                        choices.append({
                            "block_index": block_index,
                            "parent": parent, "role_partition": role,
                            "angular_partition": angular, "Lambda": int(Lambda),
                            "role_copy": role_copy, "angular_copy": angular_copy,
                            "kronecker_copy": kronecker_copy,
                        })
    return tuple(choices)


def ordered_role_cauchy_count(request):
    """Count every local Kronecker, global Young, and angular copy."""

    from ye3t.global_coupler import _typed_angular_paths
    from ye3t.representations.generalized_irreps import Partition
    from ye3t.representations.young_orthogonal import (
        young_nary_induced_multiplicity_by_character,
    )

    parent = tuple(request["target"]["young_partition"])
    output_L = int(request["target"]["L"])
    choices = tuple(_local_choices(request, index)
                    for index in range(len(request["block_sizes"])))
    routes = []
    for selected in product(*choices):
        local_parents = tuple(item["parent"] for item in selected)
        young_copies = int(young_nary_induced_multiplicity_by_character(
            tuple(Partition(item) for item in local_parents), Partition(parent)
        ))
        if not young_copies:
            continue
        Lambdas = tuple(item["Lambda"] for item in selected)
        angular_paths = _typed_angular_paths(Lambdas, output_L, "balanced")
        for gamma in range(young_copies):
            for beta, tree in enumerate(angular_paths):
                routes.append({
                    "local_choices": selected,
                    "local_copy_gauge": ORDERED_ROLE_COPY_GAUGE,
                    "block_parents": local_parents,
                    "block_Lambdas": Lambdas,
                    "young_copy": gamma,
                    "angular_copy": beta,
                    "angular_tree": tree,
                })
    routes.sort(key=lambda row: (
        row["block_parents"], row["block_Lambdas"],
        tuple((item["role_partition"], item["angular_partition"],
               item["role_copy"], item["angular_copy"], item["kronecker_copy"])
              for item in row["local_choices"]),
        row["angular_copy"], row["young_copy"],
    ))
    labels = tuple({**row, "multiplicity_index": index}
                   for index, row in enumerate(routes))
    tableau_count = int(Partition(parent).dimension)
    return {
        "schema": "ye3t_ordered_role_cauchy_count_v2",
        "request": request,
        "labels": labels,
        "multiplet_count": len(labels),
        "tableau_count": tableau_count,
        "component_count": len(labels) * tableau_count * (2 * output_L + 1),
        "validation_report": {"passed": True,
                              "source": "local_Kronecker_x_global_LR_x_binary_CG"},
    }


def _young_table(local_parents, target, bracketing):
    from ye3t.global_coupler import (
        _build_young_subgroup_specht_coupling, standard_tableaux,
    )

    rank = sum(sum(item) for item in local_parents)
    if target == (rank,) or target == (1,) * rank:
        character = "symmetric" if target == (rank,) else "antisymmetric"
        expected = tuple((sum(item),) if character == "symmetric"
                         else (1,) * sum(item) for item in local_parents)
        if local_parents != expected:
            raise ArithmeticError("Character route has incompatible local Young types.")
        return {"local_parents": local_parents,
                "analytic_character": character,
                "local_tableau_dim": 1, "target_tableau_dim": 1,
                "gamma_count": 1}, None
    dimensions = tuple(len(standard_tableaux(item)) for item in local_parents)
    local_dim = math.prod(dimensions)
    target_dim = len(standard_tableaux(target))
    matrix_units = all(sum(item) == 1 for item in local_parents)
    if matrix_units:
        gamma_count = target_dim
        scale = math.sqrt(target_dim / math.factorial(rank))
        restricted = ((tuple(scale if col == gamma else 0.0
                             for gamma in range(gamma_count)
                             for col in range(target_dim)),)
                      if rank <= 4 else None)
    else:
        restricted, gamma_count = direct_restricted_young_values(
            local_parents, target
        )
    if (local_dim < 1 or gamma_count < 1 or restricted is not None and (
        len(restricted) != local_dim or any(
            len(row) != gamma_count * target_dim for row in restricted
        )
    )):
        raise ArithmeticError("Direct ordered Cauchy Young axes are invalid.")
    if rank <= 4:
        reference = _build_young_subgroup_specht_coupling(
            local_parents, target, bracketing=bracketing
        )
        full = reference.coefficient_matrix()
        if (not reference.validation.passed
                or full.cols != gamma_count * target_dim
                or any(abs(float(full[row, col]) - restricted[row][col]) > 1e-10
                       for row in range(local_dim) for col in range(full.cols))):
            raise ArithmeticError("Direct ordered Cauchy Young map disagrees with exact orbit reference.")
    return {
        "local_parents": local_parents,
        "analytic_character": None,
        "local_tableau_dim": local_dim,
        "target_tableau_dim": target_dim,
        "gamma_count": gamma_count,
        "target_partition": target,
        **({"restricted_kind": "matrix_units", "restricted_scale": scale}
           if matrix_units else {"restricted_values": restricted}),
        "adjacent_generators": adjacent_generator_columns(target),
        "coset_transport_residual": 0.0,
        "transport_validation": "exact_subgroup_generators_and_reduced_gram",
        "storage": "identity_coset_plus_sparse_adjacent_generators_v1",
    }, None


def compile_ordered_role_cauchy(request):
    """Compile local metric duals, Young maps, and binary CG nodes."""

    from .covariant_cauchy import compile_slot_resolved_cauchy_block
    from .factorized_typed import _cg_tree_payload
    from . import lifted_cauchy_scalar as scalar

    report = ordered_role_cauchy_count(request)
    if not report["labels"]:
        raise ValueError("Ordered role Cauchy target has zero multiplicity.")
    parent = tuple(request["target"]["young_partition"])
    blocks = []
    offset = 0
    for index, size in enumerate(request["block_sizes"]):
        blocks.append({"channel": index, "size": int(size),
                       "positions": tuple(range(offset, offset + int(size)))})
        offset += int(size)
    local_tables = []
    local_index = {}
    young_tables = []
    young_index = {}
    angular_trees = []
    angular_index = {}
    routes = []
    common_cosets = None
    for label in report["labels"]:
        local_indices = []
        for choice in label["local_choices"]:
            block_index = choice["block_index"]
            key = (block_index, choice["parent"], choice["role_partition"],
                   choice["angular_partition"], choice["Lambda"])
            if key not in local_index:
                size = int(request["block_sizes"][block_index])
                ell = int(request["channels"][block_index]["l"])
                use_network = (int(request["role_dimension"]) * (2 * ell + 1)) ** size > 81
                local = compile_slot_resolved_cauchy_block(
                    request["role_dimension"], size,
                    choice["parent"], choice["role_partition"],
                    choice["angular_partition"],
                    ell, choice["Lambda"], factorized_only=use_network,
                )
                if local["multiplicity"] < 1 or not local["validation_report"]["passed"]:
                    raise ArithmeticError("Local ordered Cauchy map is invalid.")
                if use_network:
                    families = tuple(local["family_order"])
                    local_payload = {
                        "strategy": "role_angular_network",
                        "role_values": local["role_values"],
                        "angular_values": local["angular_values"],
                        "orthogonal_kernel": local["orthogonal_kernel"],
                        "role_dimension": local["role_dimension"],
                        "angular_width": local["angular_width"],
                        "factor_count": local["factor_count"],
                        "magnetic_width": local["magnetic_width"],
                        "entries_by_family": (),
                    }
                else:
                    families = tuple(dict.fromkeys(
                        coordinate[:3] for coordinate in local["analysis_vectors"]
                    ))
                    entries_by_family = {}
                    Lambda = int(choice["Lambda"])
                    for family in families:
                        terms = []
                        for tableau in range(local["parent_dimension"]):
                            for M in range(-Lambda, Lambda + 1):
                                vector = local["orthonormal_vectors"][
                                    (*family, tableau, M)
                                ]
                                column = tableau * (2 * Lambda + 1) + M + Lambda
                                for row, coefficient in enumerate(vector):
                                    if coefficient == 0:
                                        continue
                                    state = local["joint_states"][row]
                                    indices = tuple(
                                        int(role) * (2 * ell + 1) + int(m) + ell
                                        for role, m in state
                                    )
                                    terms.append((indices, column, float(coefficient)))
                        entries_by_family[family] = tuple(terms)
                    local_payload = {
                        "strategy": "sparse_monomial",
                        "entries_by_family": tuple(
                            {"family": family, "terms": terms}
                            for family, terms in entries_by_family.items()
                        ),
                    }
                local_index[key] = len(local_tables)
                local_tables.append({
                    "key": key,
                    "parent_dimension": local["parent_dimension"],
                    "copy_gauge": ORDERED_ROLE_COPY_GAUGE,
                    "family_order": families,
                    "copy_gram": scalar._exact_matrix_payload(local["copy_gram"]),
                    "validation_report": local["validation_report"],
                    **local_payload,
                })
            local_indices.append(local_index[key])
        young_key = label["block_parents"]
        if young_key not in young_index:
            young, cosets = _young_table(young_key, parent, "balanced")
            if cosets is not None:
                if common_cosets is None:
                    common_cosets = cosets
                elif common_cosets != cosets:
                    raise ArithmeticError("Young maps disagree on coset order.")
            young_index[young_key] = len(young_tables)
            young_tables.append(young)
        angular_key = (label["block_Lambdas"], label["angular_tree"])
        if angular_key not in angular_index:
            angular_index[angular_key] = len(angular_trees)
            angular_trees.append(_cg_tree_payload(label["angular_tree"]))
        routes.append({"multiplicity_index": label["multiplicity_index"],
                       "local_indices": tuple(local_indices),
                       "local_copy_indices": tuple(
                           local_tables[index]["family_order"].index((
                               choice["role_copy"], choice["angular_copy"],
                               choice["kronecker_copy"],
                           ))
                           for index, choice in zip(local_indices,
                                                    label["local_choices"], strict=True)
                       ),
                       "young_index": young_index[young_key],
                       "angular_index": angular_index[angular_key],
                       "local_choices": label["local_choices"],
                       "young_copy": label["young_copy"]})
    orbit_count = math.factorial(sum(request["block_sizes"]))
    for size in request["block_sizes"]:
        orbit_count //= math.factorial(size)
    payload = {
        "schema": "ye3t_ordered_role_cauchy_compiled_v2",
        "request": request,
        "labels": report["labels"],
        "multiplet_count": report["multiplet_count"],
        "tableau_count": report["tableau_count"],
        "component_count": report["component_count"],
        "blocks": tuple(blocks),
        "factor_types": tuple(index for index, size in enumerate(request["block_sizes"])
                              for _ in range(size)),
        "orbit_count": orbit_count,
        "coset_representatives": common_cosets,
        "local_tables": tuple(local_tables),
        "young_tables": tuple(young_tables),
        "angular_trees": tuple(angular_trees),
        "routes": tuple(routes),
        "validation_report": {
            "passed": True,
            "all_multiplicity_coordinates": len(routes) == report["multiplet_count"],
            "local_full_copy_orthonormal": True,
            "young_subduction_validated": True,
            "binary_CG_paths": True,
            "full_joint_matrix_not_materialized": True,
            "full_young_orbit_matrix_not_stored": True,
        },
    }
    payload["self_hash"] = _payload_hash(payload)
    return payload


def _payload_hash(payload):
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def validate_ordered_role_cauchy(compiled):
    """Check the complete serialized schedule before evaluating it."""

    if compiled.get("schema") != "ye3t_ordered_role_cauchy_compiled_v2":
        raise ValueError("Unknown ordered-role Cauchy artifact schema.")
    if _payload_hash(compiled) != compiled.get("self_hash"):
        raise ValueError("Ordered-role Cauchy artifact hash mismatch.")
    report = ordered_role_cauchy_count(compiled["request"])
    if (json.dumps(compiled["labels"], sort_keys=True) !=
            json.dumps(report["labels"], sort_keys=True)
            or int(compiled["multiplet_count"]) != report["multiplet_count"]
            or int(compiled["tableau_count"]) != report["tableau_count"]
            or int(compiled["component_count"]) != report["component_count"]):
        raise ValueError("Ordered-role Cauchy multiplicity labels are invalid.")
    routes = tuple(compiled["routes"])
    if (len(routes) != int(compiled["multiplet_count"])
            or tuple(int(item["multiplicity_index"]) for item in routes)
            != tuple(range(len(routes)))
            or not compiled["validation_report"].get("passed", False)):
        raise ValueError("Ordered-role Cauchy route inventory is invalid.")
    sizes = tuple(int(value) for value in compiled["request"]["block_sizes"])
    expected_types = tuple(index for index, size in enumerate(sizes)
                           for _ in range(size))
    expected_blocks = []
    offset = 0
    for index, size in enumerate(sizes):
        expected_blocks.append({"channel": index, "size": size,
                                "positions": tuple(range(offset, offset + size))})
        offset += size
    if (tuple(compiled["factor_types"]) != expected_types
            or json.dumps(compiled["blocks"], sort_keys=True) !=
            json.dumps(expected_blocks, sort_keys=True)):
        raise ValueError("Ordered-role Cauchy factor axes are invalid.")
    orbit_count = math.factorial(len(expected_types))
    for size in sizes:
        orbit_count //= math.factorial(size)
    if int(compiled["orbit_count"]) != orbit_count:
        raise ValueError("Ordered-role Cauchy orbit count is invalid.")
    cosets = compiled["coset_representatives"]
    if cosets is not None and (len(cosets) != orbit_count or any(
        sorted(row) != list(range(len(expected_types))) for row in cosets
    )):
        raise ValueError("Ordered-role Cauchy coset axes are invalid.")
    for local in compiled["local_tables"]:
        if local.get("copy_gauge") != ORDERED_ROLE_COPY_GAUGE:
            raise ValueError("Local Cauchy copy gauge or family order is invalid.")
        if local.get("strategy") == "sparse_monomial":
            families = tuple(tuple(row["family"]) for row in local["entries_by_family"])
        elif local.get("strategy") == "role_angular_network":
            families = tuple(tuple(row) for row in local["family_order"])
            if (len(local["orthogonal_kernel"]) != len(families)
                    or local["factor_count"] != sum(local["key"][1])):
                raise ValueError("Factorized local Cauchy kernel axes are invalid.")
        else:
            raise ValueError("Unknown ordered-role local strategy.")
        if tuple(tuple(row) for row in local.get("family_order", ())) != families:
            raise ValueError("Local Cauchy family order is invalid.")
        if not local["validation_report"].get("full_copy_orthonormal_exact", False):
            raise ValueError("Local Cauchy copy orthonormalization is missing.")
    if any(
        len(route["local_indices"]) != len(sizes)
        or len(route["local_choices"]) != len(sizes)
        or route["young_index"] not in range(len(compiled["young_tables"]))
        or route["angular_index"] not in range(len(compiled["angular_trees"]))
        or any(index not in range(len(compiled["local_tables"]))
               for index in route["local_indices"])
        for route in routes
    ):
        raise ValueError("Ordered-role Cauchy route table indices are invalid.")
    request_json = json.dumps(compiled["request"], sort_keys=True,
                              allow_nan=False, separators=(",", ":"))
    if any("values" in young for young in compiled["young_tables"]):
        # The v2 pre-compact table is still readable. Its hash and every route,
        # local map, and Young coefficient must match the current compiler.
        import numpy as np
        from ye3t.representations.young_orthogonal import (
            _young_subgroup_shuffle_representatives,
        )

        current = _recompiled_payload(request_json)
        if tuple(tuple(row) for row in cosets) != _young_subgroup_shuffle_representatives(sizes):
            raise ValueError("Legacy ordered-role coset order differs from the canonical source.")
        for key in ("labels", "blocks", "factor_types", "local_tables",
                    "angular_trees", "routes"):
            if json.dumps(compiled[key], sort_keys=True) != json.dumps(current[key], sort_keys=True):
                raise ValueError("Legacy ordered-role Cauchy schedule differs from compiler replay.")
        for old, new in zip(compiled["young_tables"], current["young_tables"], strict=True):
            if "values" not in old:
                if json.dumps(old, sort_keys=True) != json.dumps(new, sort_keys=True):
                    raise ValueError("Legacy ordered-role Young character differs from compiler replay.")
                continue
            if (json.dumps(old["local_parents"]) != json.dumps(new["local_parents"])
                    or old["gamma_count"] != new["gamma_count"]
                    or not np.allclose(np.asarray(old["values"], dtype=np.float64),
                                       expanded_young_values(new, cosets),
                                       atol=1e-8, rtol=1e-8)):
                raise ValueError("Legacy ordered-role Young coefficients differ from compiler replay.")
    elif _recompiled_hash(request_json) != compiled["self_hash"]:
        raise ValueError("Ordered-role Cauchy coefficients failed compiler replay.")
    return True


@lru_cache(maxsize=32)
def _recompiled_hash(request_json):
    return compile_ordered_role_cauchy(json.loads(request_json))["self_hash"]


@lru_cache(maxsize=32)
def _recompiled_payload(request_json):
    return compile_ordered_role_cauchy(json.loads(request_json))


def _local_analysis(local, factors, dtype, device, bound=None):
    import torch

    tableau_dim = int(local["parent_dimension"])
    Lambda = int(local["key"][-1])
    batch_shape = factors[0].shape[:-2]
    if local["strategy"] == "role_angular_network":
        prepared = None if bound is None else bound.local[id(local)]
        role = (prepared["role"] if prepared is not None else torch.as_tensor(
            local["role_values"], dtype=dtype, device=device,
        ))
        angular = (prepared["angular"] if prepared is not None else torch.as_tensor(
            local["angular_values"], dtype=dtype, device=device,
        ))
        kernel = (prepared["kernel"] if prepared is not None else torch.as_tensor(
            local["orthogonal_kernel"], dtype=dtype, device=device,
        ))
        count = int(local["factor_count"])
        role_dim = int(local["role_dimension"])
        angular_width = int(local["angular_width"])
        role_columns = int(role.shape[0])
        partial = role.expand(*batch_shape, *role.shape)
        for factor_index, factor in enumerate(factors):
            remaining = count - factor_index - 1
            partial = partial.reshape(
                *batch_shape, role_columns, role_dim,
                role_dim ** remaining, angular_width ** factor_index,
            )
            partial = torch.einsum("...iarm,...av->...irmv", partial, factor)
            partial = partial.reshape(
                *batch_shape, role_columns, role_dim ** remaining,
                angular_width ** (factor_index + 1),
            )
        partial = partial.reshape(*batch_shape, role_columns, angular_width ** count)
        role_angular = torch.einsum("...im,jmM->...ijM", partial, angular)
        return torch.einsum("...ijM,cijt->...ctM", role_angular, kernel.conj())
    family_count = len(local["entries_by_family"])
    family_width = tableau_dim * (2 * Lambda + 1)
    width = family_count * family_width
    terms = tuple(
        (state, family_index * family_width + int(column), coefficient)
        for family_index, row in enumerate(local["entries_by_family"])
        for state, column, coefficient in row["terms"]
    ) if bound is None else None
    prepared = None if bound is None else bound.local[id(local)]
    term_count = len(terms) if prepared is None else int(prepared["columns"].numel())
    if not term_count:
        return torch.zeros((*batch_shape, family_count, tableau_dim, 2 * Lambda + 1),
                           dtype=dtype, device=device)
    indices = (prepared["indices"] if prepared is not None else
               torch.as_tensor([row[0] for row in terms],
                               dtype=torch.long, device=device))
    columns = (prepared["columns"] if prepared is not None else
               torch.as_tensor([row[1] for row in terms],
                               dtype=torch.long, device=device))
    coefficients = (prepared["coefficients"] if prepared is not None else
                    torch.as_tensor([row[2] for row in terms],
                                    dtype=dtype, device=device))
    product_values = torch.ones((*batch_shape, term_count),
                                dtype=dtype, device=device)
    for factor_index, factor in enumerate(factors):
        flat = factor.reshape(*batch_shape, -1)
        product_values = product_values * flat[..., indices[:, factor_index]]
    output = torch.zeros((*batch_shape, width), dtype=dtype, device=device)
    output = output.scatter_add(
        -1, columns.expand(*batch_shape, -1),
        product_values * coefficients.conj(),
    )
    return output.reshape(*batch_shape, family_count, tableau_dim, 2 * Lambda + 1)


def _one_block_analysis(compiled, factors, dtype, device, bound):
    """Contract all copies of a single repeated block in grouped tensors."""

    import torch

    routes = compiled["routes"]
    if (len(compiled["blocks"]) != 1 or compiled["orbit_count"] != 1
            or any(route["young_copy"] != 0 for route in routes)
            or any(compiled["angular_trees"][route["angular_index"]]["kind"]
                   != "leaf" for route in routes)):
        return None
    local_order = tuple(dict.fromkeys(int(route["local_indices"][0])
                                      for route in routes))
    parts = []
    positions = {}
    offset = 0
    for local_index in local_order:
        local = compiled["local_tables"][local_index]
        part = _local_analysis(local, factors, dtype, device, bound)
        young_indices = {route["young_index"] for route in routes
                         if int(route["local_indices"][0]) == local_index}
        if len(young_indices) != 1:
            raise ArithmeticError("One repeated block has inconsistent Young maps.")
        young = compiled["young_tables"][next(iter(young_indices))]
        if young["analytic_character"] is None:
            rows = int(young["local_tableau_dim"])
            columns = int(young["target_tableau_dim"])
            matrix = transported_young_slice(
                young, tuple(range(sum(compiled["request"]["block_sizes"]))),
                0, dtype, device,
                prepared=None if bound is None else bound.young[id(young)],
                coset_index=0,
            )
            part = torch.einsum("...cjm,jt->...ctm", part, matrix.conj())
        elif young["analytic_character"] not in {"symmetric", "antisymmetric"}:
            raise ArithmeticError("Unknown analytic Young character.")
        for family_index in range(part.shape[-3]):
            positions[(local_index, family_index)] = offset + family_index
        offset += part.shape[-3]
        parts.append(part)
    indices = (bound.one_block_indices if bound is not None else torch.as_tensor([
        positions[(int(route["local_indices"][0]),
                   int(route["local_copy_indices"][0]))]
        for route in routes
    ], dtype=torch.long, device=device))
    return torch.cat(parts, dim=-3).index_select(-3, indices)


def evaluate_ordered_role_cauchy(compiled, values, *, upstream=None,
                                 basis="real_tesseral", factor_types=None,
                                 bound=None):
    """Return every ``(a,t,M)`` coordinate and an optional factor VJP."""

    import numpy as np
    import torch

    from . import lifted_cauchy_scalar as scalar
    from .factorized_typed import _evaluate_cg_tree

    if (compiled.get("schema") != "ye3t_ordered_role_cauchy_compiled_v2"
            or not compiled.get("validation_report", {}).get("passed", False)):
        raise ValueError("Ordered-role Cauchy schedule must be validated at loading.")
    if basis not in {"real_tesseral", "complex_condon_shortley"}:
        raise ValueError("Unknown Cauchy magnetic basis.")
    canonical_types = tuple(compiled["factor_types"])
    factor_types = (canonical_types if factor_types is None else
                    tuple(int(value) for value in factor_types))
    if sorted(factor_types) != sorted(canonical_types):
        raise ValueError("factor_types must permute the compiled channel word.")
    is_tensor = isinstance(values, torch.Tensor)
    if is_tensor:
        if values.ndim < 3 or len({int(channel["l"]) for channel in
                                   compiled["request"]["channels"]}) != 1:
            raise ValueError("Tensor input needs a common l and (...,N,d,2l+1).")
        if values.shape[-3] != len(factor_types):
            raise ValueError("Ordered factor axis has the wrong length.")
        factors = tuple(values[..., index, :, :]
                        for index in range(len(factor_types)))
    else:
        factors = tuple(values)
    if len(factors) != len(factor_types):
        raise ValueError("Supply one role-by-magnetic multiplet per factor.")
    numpy_input = not isinstance(factors[0], torch.Tensor)
    original = tuple(torch.as_tensor(item) for item in factors)
    batch_shape = original[0].shape[:-2]
    role_dim = int(compiled["request"]["role_dimension"])
    for item, channel in zip(original, factor_types, strict=True):
        ell = int(compiled["request"]["channels"][channel]["l"])
        if item.shape[:-2] != batch_shape or item.shape[-2:] != (
            role_dim, 2 * ell + 1
        ):
            raise ValueError("Each ordered factor needs [role_dimension,2l+1].")
    if any(item.dtype != original[0].dtype or item.device != original[0].device
           for item in original):
        raise ValueError("Ordered factors must share dtype and device.")
    if bound is not None and (bound.compiled is not compiled
                              or bound.input_dtype != original[0].dtype
                              or bound.device != original[0].device
                              or bound.basis != basis):
        raise ValueError("Bound Cauchy tensors do not match artifact, dtype, device, and basis.")
    if not (original[0].dtype.is_floating_point or original[0].dtype.is_complex):
        raise TypeError("Ordered factors require floating or complex values.")
    if upstream is not None:
        original = tuple(item if item.requires_grad else
                         item.detach().clone().requires_grad_(True)
                         for item in original)
    if basis == "real_tesseral":
        if any(item.is_complex() and bool(torch.any(item.imag.abs() > 1e-13))
               for item in original):
            raise ValueError("Real-tesseral factors must be real.")
        dtype = (torch.complex128 if original[0].dtype in
                 {torch.float64, torch.complex128} else torch.complex64)
        converted = []
        for item, channel in zip(original, factor_types, strict=True):
            ell = int(compiled["request"]["channels"][channel]["l"])
            matrix = (bound.real_form[ell] if bound is not None else
                      torch.as_tensor(np.asarray(
                          scalar._real_form_matrix(ell), dtype=np.complex128
                      ), dtype=dtype, device=item.device))
            converted.append(item.to(dtype) @ matrix.T)
        factors = tuple(converted)
    else:
        dtype = original[0].dtype
        factors = original
    device = factors[0].device
    rep = canonical_factor_coset(factor_types, canonical_types)
    legacy_young = any("values" in young for young in compiled["young_tables"])
    coset = (next(index for index, candidate in enumerate(compiled["coset_representatives"])
                  if tuple(candidate) == rep)
             if legacy_young else None)
    one_block = _one_block_analysis(compiled, factors, dtype, device, bound)
    local_cache = {}
    angular_cache = {}
    young_cache = {}
    outputs = []
    for route in (() if one_block is not None else compiled["routes"]):
        local_outputs = []
        for block, local_index, family_index in zip(
            compiled["blocks"], route["local_indices"],
            route["local_copy_indices"], strict=True,
        ):
            key = (local_index, coset)
            if key not in local_cache:
                local_cache[key] = _local_analysis(
                    compiled["local_tables"][local_index],
                    tuple(factors[rep[position]] for position in block["positions"]),
                    dtype, device, bound,
                )
            local_outputs.append(local_cache[key][..., int(family_index), :, :])
        angular_key = (route["angular_index"], tuple(route["local_indices"]),
                       tuple((choice["role_copy"], choice["angular_copy"],
                              choice["kronecker_copy"])
                             for choice in route["local_choices"]))
        if angular_key not in angular_cache:
            angular_cache[angular_key] = _evaluate_cg_tree(
                compiled["angular_trees"][route["angular_index"]],
                local_outputs, dtype, device, bound,
            )
        angular = angular_cache[angular_key]
        young = compiled["young_tables"][route["young_index"]]
        if young["analytic_character"] is not None:
            weight = 1.0 / math.sqrt(int(compiled["orbit_count"]))
            if young["analytic_character"] == "antisymmetric":
                weight *= (-1) ** sum(rep[left] > rep[right]
                                      for left in range(len(rep))
                                      for right in range(left + 1, len(rep)))
            outputs.append(angular * weight)
            continue
        rows = int(young["local_tableau_dim"])
        columns = int(young["target_tableau_dim"])
        gamma = int(route["young_copy"])
        young_key = (int(route["young_index"]), gamma)
        if young_key not in young_cache:
            young_cache[young_key] = transported_young_slice(
                young, rep, gamma, dtype, device,
                prepared=None if bound is None else bound.young[id(young)],
                coset_index=coset,
            )
        matrix = young_cache[young_key]
        if angular.shape[-2] != rows:
            raise ArithmeticError("Local and Young tableau axes disagree.")
        outputs.append(torch.einsum("...jm,jt->...tm", angular, matrix.conj()))
    result = (one_block if one_block is not None else
              torch.stack(outputs, dim=-3))
    if basis == "real_tesseral":
        L = int(compiled["request"]["target"]["L"])
        output_form = (bound.real_form[L] if bound is not None else
                       torch.as_tensor(np.asarray(
                           scalar._real_form_matrix(L), dtype=np.complex128
                       ), dtype=dtype, device=device))
        phase = (-1j) ** ((sum(
            size * int(channel["l"]) for size, channel in zip(
                compiled["request"]["block_sizes"],
                compiled["request"]["channels"], strict=True
            )) - L) % 2)
        projector = phase * output_form.conj().T
        result = result @ projector.T
        scale = max(1.0, float(result.detach().abs().max()))
        if float(result.detach().imag.abs().max()) > 5e-11 * scale:
            raise ArithmeticError("Ordered Cauchy real form has a complex residual.")
        result = result.real
    gradients = tuple(torch.zeros_like(item) for item in original)
    if upstream is not None:
        seed = torch.as_tensor(upstream, dtype=result.dtype, device=device)
        if seed.shape != result.shape:
            raise ValueError("upstream must match the full (a,t,M) output shape.")
        gradients = torch.autograd.grad(
            result, original, grad_outputs=seed.conj(),
            create_graph=torch.is_grad_enabled() and not numpy_input,
        )
        gradients = tuple(item.conj() for item in gradients)
    if numpy_input:
        return (result.detach().cpu().numpy(),
                tuple(item.detach().cpu().numpy() for item in gradients))
    if is_tensor:
        return result, torch.stack(gradients, dim=-3)
    return result, gradients


class YE3TOrderedRoleCauchyTorchRuntime:
    """Bind one complete ordered-role Cauchy artifact to a torch device.

    Mathematical contract: the exact orthonormal local copy maps, Young
    subduction, and binary angular maps retain every ``(a,t,M)`` coordinate.
    Inputs: a validated compiled artifact, input dtype, device, and magnetic
    basis. Outputs: repeated differentiable evaluation of ordered factor
    batches. Does not: aggregate neighbors or physical motif occurrences.
    """

    def __init__(self, compiled, *, dtype, device,
                 basis="complex_condon_shortley"):
        import numpy as np
        import torch

        from . import lifted_cauchy_scalar as scalar
        from .factorized_typed import _bind_cg_trees

        validate_ordered_role_cauchy(compiled)
        if basis not in {"real_tesseral", "complex_condon_shortley"}:
            raise ValueError("Unknown Cauchy magnetic basis.")
        self.compiled = compiled
        self.input_dtype = dtype
        self.device = torch.device(device)
        self.basis = basis
        if basis == "real_tesseral":
            self.dtype = (torch.complex128 if dtype in
                          {torch.float64, torch.complex128} else torch.complex64)
        else:
            self.dtype = dtype
        self.local = {}
        for local in compiled["local_tables"]:
            if local["strategy"] == "role_angular_network":
                self.local[id(local)] = {
                    "role": torch.as_tensor(local["role_values"],
                                             dtype=self.dtype, device=self.device),
                    "angular": torch.as_tensor(local["angular_values"],
                                                dtype=self.dtype, device=self.device),
                    "kernel": torch.as_tensor(local["orthogonal_kernel"],
                                               dtype=self.dtype, device=self.device),
                }
                continue
            family_width = int(local["parent_dimension"]) * (
                2 * int(local["key"][-1]) + 1
            )
            terms = tuple(
                (state, family_index * family_width + int(column), coefficient)
                for family_index, row in enumerate(local["entries_by_family"])
                for state, column, coefficient in row["terms"]
            )
            self.local[id(local)] = {
                "indices": torch.as_tensor(
                    [row[0] for row in terms], dtype=torch.long,
                    device=self.device,
                ),
                "columns": torch.as_tensor(
                    [row[1] for row in terms], dtype=torch.long,
                    device=self.device,
                ),
                "coefficients": torch.as_tensor(
                    [row[2] for row in terms], dtype=self.dtype,
                    device=self.device,
                ),
            }
        self.cg = _bind_cg_trees(compiled["angular_trees"],
                                 self.dtype, self.device)
        self.one_block_indices = None
        routes = compiled["routes"]
        if (len(compiled["blocks"]) == 1 and compiled["orbit_count"] == 1
                and all(route["young_copy"] == 0 for route in routes)
                and all(compiled["angular_trees"][route["angular_index"]]["kind"]
                        == "leaf" for route in routes)):
            local_order = tuple(dict.fromkeys(int(route["local_indices"][0])
                                              for route in routes))
            positions = {}
            offset = 0
            for local_index in local_order:
                local = compiled["local_tables"][local_index]
                for family_index in range(len(local["family_order"])):
                    positions[(local_index, family_index)] = offset + family_index
                offset += len(local["family_order"])
            self.one_block_indices = torch.as_tensor([
                positions[(int(route["local_indices"][0]),
                           int(route["local_copy_indices"][0]))]
                for route in routes
            ], dtype=torch.long, device=self.device)
        self.young = {}
        for young in compiled["young_tables"]:
            if young["analytic_character"] is not None:
                continue
            self.young[id(young)] = bind_young_restriction(
                young, self.dtype, self.device,
            )
        self.real_form = {}
        if basis == "real_tesseral":
            angular_momenta = {
                int(channel["l"]) for channel in compiled["request"]["channels"]
            } | {int(compiled["request"]["target"]["L"])}
            for ell in angular_momenta:
                self.real_form[ell] = torch.as_tensor(np.asarray(
                    scalar._real_form_matrix(ell), dtype=np.complex128,
                ), dtype=self.dtype, device=self.device)

    def evaluate(self, factors, *, upstream=None, factor_types=None):
        """Evaluate all coupled coordinates of an ordered torch factor batch."""

        return evaluate_ordered_role_cauchy(
            self.compiled, factors, upstream=upstream,
            basis=self.basis, factor_types=factor_types, bound=self,
        )


def bind_ordered_role_cauchy_torch(compiled, *, dtype, device,
                                  basis="complex_condon_shortley"):
    """Build a reusable tensor runtime for a compiled ordered Cauchy artifact."""

    return YE3TOrderedRoleCauchyTorchRuntime(
        compiled, dtype=dtype, device=device, basis=basis,
    )


def lower_ordered_role_cauchy_execution_plan(compiled):
    """Put a source-neutral Cauchy factor schedule in the core plan format."""
    from ye3t.couplings import compile_execution_plan
    from ye3t.execution_plan import (
        YE3TCarrierKey, YE3TCarrierLayout, YE3TRuntimeInstruction,
        YE3T_O3_PRIMARY_CONVENTION,
    )

    validate_ordered_role_cauchy(compiled)
    request = compiled["request"]
    parent = request["target"]
    factors = tuple(YE3TCarrierKey(
        rank=1, partition=(1,), rotation_L=int(channel["l"]),
        parity=int(channel.get("parity", (-1) ** int(channel["l"]))),
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
    ) for channel, size in zip(request["channels"], request["block_sizes"], strict=True)
        for _ in range(int(size)))
    output = YE3TCarrierKey(
        rank=sum(request["block_sizes"]),
        partition=tuple(parent["young_partition"]),
        rotation_L=int(parent["L"]), parity=int(parent["o3_parity"]),
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
    )
    layout = YE3TCarrierLayout(
        key=output, channel_count=int(compiled["multiplet_count"]),
        tableau_count=int(compiled["tableau_count"]),
        magnetic_count=2 * int(parent["L"]) + 1,
    )
    instruction = YE3TRuntimeInstruction(
        instruction_id="ordered_role_cauchy", opcode="ordered_role_cauchy_factorized",
        input_carriers=factors, output_carrier=output,
        metadata={"schema": "ye3t_ordered_role_cauchy_execution_v1",
                  "compiled": compiled, "compiler_hash": compiled["self_hash"],
                  "factor_axis_order": "ordered_factor_role_m",
                  "output_axis_order": ("a", "t", "M")},
    )
    return compile_execution_plan(
        carrier_layouts=(layout,), instructions=(instruction,),
        forward_schedule=(instruction.instruction_id,),
        reverse_schedule=(instruction.instruction_id,),
        convention_id=YE3T_O3_PRIMARY_CONVENTION,
        certificate={"passed": True, "scope": "source_neutral_ordered_role_factors"},
        provenance={"compiler_owner": "ye3t",
                    "source": "ordered_role_cauchy_factorized",
                    "compiled_hash": compiled["self_hash"]},
    )


def bind_ordered_role_execution_plan_torch(plan, *, dtype, device,
                                           basis="complex_condon_shortley"):
    """Bind a core Cauchy plan for repeated factor evaluations and gradients."""
    from ye3t.execution_plan import YE3TExecutionPlan

    if not isinstance(plan, YE3TExecutionPlan):
        plan = YE3TExecutionPlan.from_dict(plan)
    if (len(plan.instructions) != 1
            or plan.instructions[0].opcode != "ordered_role_cauchy_factorized"):
        raise ValueError("Execution plan has no ordered role Cauchy factor instruction.")
    return bind_ordered_role_cauchy_torch(
        plan.instructions[0].metadata["compiled"],
        dtype=dtype, device=device, basis=basis,
    )
