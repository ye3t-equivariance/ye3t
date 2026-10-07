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
    coupling = _build_young_subgroup_specht_coupling(
        local_parents, target, bracketing=bracketing
    )
    if not coupling.validation.passed:
        raise ArithmeticError("Ordered Cauchy Young subduction failed validation.")
    matrix = coupling.coefficient_matrix()
    cosets = tuple(tuple(int(value) for value in row)
                   for row in coupling.tensor.coset_reps)
    dimensions = tuple(len(standard_tableaux(item)) for item in local_parents)
    local_dim = math.prod(dimensions)
    target_dim = len(coupling.tensor.target_tableaux)
    gamma_count = len(coupling.tensor.vectors) // target_dim
    row_lookup = {
        (int(item.coset_index), tuple(item.child_tableau_indices)): index
        for index, item in enumerate(coupling.tensor.induced_basis)
    }
    column_lookup = {
        (int(item.rho), int(item.target_tableau_index)): index
        for index, item in enumerate(coupling.tensor.vectors)
    }
    canonical_rows = tuple(
        row_lookup[(coset, tableaux)]
        for coset in range(len(cosets))
        for tableaux in product(*(range(value) for value in dimensions))
    )
    canonical_columns = tuple(
        column_lookup[(gamma, tableau)]
        for gamma in range(gamma_count)
        for tableau in range(target_dim)
    )
    if (len(row_lookup) != matrix.rows or len(column_lookup) != matrix.cols
            or len(canonical_rows) != matrix.rows
            or len(canonical_columns) != matrix.cols):
        raise ArithmeticError("Young basis metadata omits matrix coordinates.")
    return {
        "local_parents": local_parents,
        "analytic_character": None,
        "local_tableau_dim": local_dim,
        "target_tableau_dim": target_dim,
        "gamma_count": gamma_count,
        "values": tuple(tuple(float(matrix[row, column])
                              for column in canonical_columns)
                        for row in canonical_rows),
    }, cosets


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
            "full_orbit_matrix_not_materialized": True,
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
    if _recompiled_hash(request_json) != compiled["self_hash"]:
        raise ValueError("Ordered-role Cauchy coefficients failed compiler replay.")
    return True


@lru_cache(maxsize=32)
def _recompiled_hash(request_json):
    return compile_ordered_role_cauchy(json.loads(request_json))["self_hash"]


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
            matrix = (bound.young[id(young)][0, :, 0, :] if bound is not None else
                      torch.as_tensor([
                          row[:columns] for row in young["values"][:rows]
                      ], dtype=dtype, device=device))
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
    if compiled["young_tables"][0]["analytic_character"] is not None:
        rep = tuple(index for channel in dict.fromkeys(canonical_types)
                    for index, candidate in enumerate(factor_types)
                    if candidate == channel)
        coset = 0
    else:
        coset = next((index for index, candidate in enumerate(
            compiled["coset_representatives"]
        ) if all(factor_types[candidate[position]] == canonical_types[position]
                 for position in range(len(candidate)))), None)
        if coset is None:
            raise ArithmeticError("No Cauchy coset transports the factor word.")
        rep = compiled["coset_representatives"][coset]
    one_block = _one_block_analysis(compiled, factors, dtype, device, bound)
    local_cache = {}
    angular_cache = {}
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
        matrix = (
            bound.young[id(young)][coset, :, gamma, :]
            if bound is not None else torch.as_tensor([
                row[gamma * columns:(gamma + 1) * columns]
                for row in young["values"][coset * rows:(coset + 1) * rows]
            ], dtype=dtype, device=device)
        )
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
            rows = int(young["local_tableau_dim"])
            columns = int(young["target_tableau_dim"])
            gamma = int(young["gamma_count"])
            self.young[id(young)] = torch.as_tensor(
                young["values"], dtype=self.dtype, device=self.device,
            ).reshape(int(compiled["orbit_count"]), rows, gamma, columns)
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
