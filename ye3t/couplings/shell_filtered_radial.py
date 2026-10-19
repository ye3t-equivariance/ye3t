"""Frozen shell-filtered radial source family for lifted-density carriers.

The source functions multiply one compiler-owned origin-regular radial family
by fixed shell windows of the scaled distance ``x = r / r_cut``::

    psi_{s,n,l}(x) = w_s(x) R_{n,l}(x),  s = 0..S-1,  n = 0..nmax-1,

where ``R_{n,l}`` is the base family's envelope-times-polynomial factor (the
regular-solid ``x^l`` stays in the angular factor) and the windows are raw
Gaussians ``exp(-(x - xi_s)^2 / (2 Delta^2))`` or their softmax normalization
over ``s``.  Windowed functions are not orthogonal.  A plan freezes one linear
transform ``T_l`` per angular degree such that ``psi~_l = T_l psi_l`` is
orthonormal under the radial-volume measure ``x^{2l} x^2 dx`` on ``[0, 1]``
(the ``x^{2l}`` carries the angular factor's radial part).  The Gram is
integrated by a Gauss--Legendre rule and checked with an independent rule;
``ordered`` uses the Cholesky factor (lower-triangular ``T_l``, nested
prefixes), ``symmetric`` the Loewdin inverse square root.  Scope
``joint_source`` mixes every ``(s, n)``; ``shell_local`` mixes within each
shell only and records the remaining cross-shell overlap.  No rank reduction,
jitter or pseudoinverse is applied: a singular Gram is a compile error.

The flattened radial channel index of the tagged catalogue is
``q = s * nmax + n`` (shell-major).  Evaluation applies ``T_l`` to the raw
window-times-radial values of every edge before the inclusive sum, which is
the same operation as transforming the accumulated density because the map
is linear and fixed; reverse-mode derivatives use the transpose through
autograd.  References: ``YE3T_REORTHOGONALIZED_SHELL_BASIS_HANDOFF.md``
sections 2, 4, 7 and 11; ``ye3t_standardized_mp.md`` (revised lifted
construction, formal parent ``(N)``).
"""

import hashlib
import json
import math

import numpy as np
import torch

from .chebyshev_c2 import CHEBYSHEV_C2_SOURCE_FAMILY, shifted_chebyshev_ladder_with_derivative
from .orthogonal_shifted_jacobi import (
    ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY, shifted_jacobi_ladder_with_derivative,
)


SHELL_FILTERED_SOURCE_FAMILY_PREFIX = "shell_filtered_"
SHELL_FILTERED_SOURCE_PLAN_SCHEMA = "ye3t_shell_filtered_radial_source_plan_v1"
SHELL_FILTERED_BASE_FAMILIES = {
    "shell_filtered_chebyshev_c2_v1": CHEBYSHEV_C2_SOURCE_FAMILY,
    "shell_filtered_shifted_jacobi_v1": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
}
_ENVELOPE_POWER = {CHEBYSHEV_C2_SOURCE_FAMILY: 3, ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY: 2}
_ZERO_EXTENSION = {CHEBYSHEV_C2_SOURCE_FAMILY: "C2", ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY: "C1"}
_SHELL_KINDS = ("softmax_gaussian", "gaussian")
_SCOPES = ("joint_source", "shell_local")
_CONVENTIONS = {"cholesky": "ordered", "lowdin": "symmetric"}
_MEASURE = "radial_volume_scaled"


def _stable_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def is_shell_filtered_family(family_id):
    return isinstance(family_id, str) and family_id.startswith(SHELL_FILTERED_SOURCE_FAMILY_PREFIX)


def shell_filtered_family_name(family_id):
    """Configured family name without the plan-hash suffix."""
    name = str(family_id).split(":", 1)[0]
    if name not in SHELL_FILTERED_BASE_FAMILIES:
        raise ValueError("Unknown shell-filtered radial family: " + str(family_id))
    return name


def normalize_shell_filter_settings(shells, orthogonalization=None):
    """Validate and canonicalize the window and orthogonalization settings."""
    if not isinstance(shells, dict) or set(shells) != {"kind", "centers", "width"}:
        raise ValueError("shells needs exactly kind, centers and width")
    kind = str(shells["kind"])
    if kind not in _SHELL_KINDS:
        raise ValueError("shells.kind must be softmax_gaussian or gaussian")
    centers = shells["centers"]
    if (not isinstance(centers, (list, tuple)) or not centers
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in centers)):
        raise ValueError("shells.centers must be a nonempty list of numbers (fractions of the cutoff)")
    centers = tuple(float(value) for value in centers)
    if (any(not math.isfinite(value) or not 0 < value <= 1 for value in centers)
            or any(right <= left for left, right in zip(centers, centers[1:]))):
        raise ValueError("shells.centers must be strictly increasing fractions in (0, 1]")
    if isinstance(shells["width"], bool) or not isinstance(shells["width"], (int, float)):
        raise ValueError("shells.width must be a number")
    width = float(shells["width"])
    if not math.isfinite(width) or width <= 0:
        raise ValueError("shells.width must be a positive finite fraction of the cutoff")
    settings = dict(orthogonalization or {})
    unknown = set(settings) - {"scope", "factorization", "measure", "tolerance",
                               "construction_points", "check_points"}
    if unknown:
        raise ValueError("Unknown orthogonalization settings: " + ", ".join(sorted(unknown)))
    scope = str(settings.get("scope", "joint_source"))
    factorization = str(settings.get("factorization", "cholesky"))
    measure = str(settings.get("measure", _MEASURE))
    if scope not in _SCOPES or factorization not in _CONVENTIONS or measure != _MEASURE:
        raise ValueError("orthogonalization needs scope joint_source/shell_local, factorization "
                         "cholesky/lowdin and measure radial_volume_scaled")
    tolerance = float(settings.get("tolerance", 1e-8))
    points = int(settings.get("construction_points", 256))
    check_points = int(settings.get("check_points", 769))
    if not 0 < tolerance < 1e-3 or points < 8 or check_points < 8 or points == check_points:
        raise ValueError("orthogonalization tolerance must lie in (0, 1e-3) and the two quadrature rules must differ")
    return ({"kind": kind, "centers": centers, "width": width},
            {"scope": scope, "convention": _CONVENTIONS[factorization], "factorization": factorization,
             "measure": measure, "tolerance": tolerance, "construction_points": points,
             "check_points": check_points, "compile_dtype": "float64"})


def shell_windows(x, shells):
    """Window values ``w_s(x)`` with shells on the last axis."""
    centers = torch.as_tensor(shells["centers"], dtype=x.dtype, device=x.device)
    logits = -0.5 * ((x[..., None] - centers) / float(shells["width"])) ** 2
    if shells["kind"] == "softmax_gaussian":
        return torch.softmax(logits, dim=-1)
    return torch.exp(logits)


def shell_filtered_raw_values(plan, x, angular_l):
    """Raw ``psi_{s,n,l}(x)`` with envelope, shell-major rows ``[S*nmax, ...x]``."""
    base = plan["base_family"]
    count = int(plan["radial_count"])
    if base == CHEBYSHEV_C2_SOURCE_FAMILY:
        ladder = shifted_chebyshev_ladder_with_derivative(count - 1, x)[0]
    else:
        ladder = shifted_jacobi_ladder_with_derivative(count - 1, 4, 2 * int(angular_l) + 2, x)[0]
    radial = torch.stack(ladder) * (1 - x) ** _ENVELOPE_POWER[base]
    windows = torch.movedim(shell_windows(x, plan["shells"]), -1, 0)
    return (windows[:, None] * radial[None]).reshape((-1,) + tuple(x.shape))


def shell_filtered_radial_values(plan, x, angular_l, transform=None):
    """Transformed ``psi~_l = T_l psi_l`` as ``[S*nmax, ...x]``."""
    raw = shell_filtered_raw_values(plan, x, angular_l)
    if transform is None:
        transform = torch.tensor(plan["transforms"][str(int(angular_l))], dtype=x.dtype, device=x.device)
    return torch.tensordot(transform, raw, dims=1)


def _quadrature(points):
    nodes, weights = np.polynomial.legendre.leggauss(int(points))
    return (torch.tensor((nodes + 1) / 2, dtype=torch.float64),
            torch.tensor(weights / 2, dtype=torch.float64))


def _gram(plan, angular_l, points):
    nodes, weights = _quadrature(points)
    raw = shell_filtered_raw_values(plan, nodes, angular_l)
    gram = raw @ (weights * nodes ** (2 * int(angular_l) + 2) * raw).T
    return (gram + gram.T) / 2


def _block_transform(gram, factorization):
    scales = torch.sqrt(torch.diagonal(gram))
    if not bool(torch.isfinite(gram).all()) or not bool((scales > 0).all()):
        raise ArithmeticError("Zero or nonfinite source norm in the shell-filtered Gram")
    equilibrated = gram / scales[:, None] / scales[None, :]
    eigenvalues = torch.linalg.eigvalsh(equilibrated)
    condition = float(eigenvalues.max() / eigenvalues.min()) if float(eigenvalues.min()) > 0 else math.inf
    if factorization == "cholesky":
        try:
            lower = torch.linalg.cholesky(equilibrated)
        except torch.linalg.LinAlgError as error:
            raise ArithmeticError("Shell-filtered source functions are not linearly independent "
                                  "(Cholesky failed; equilibrated condition number "
                                  + repr(condition) + ")") from error
        transform = torch.linalg.solve_triangular(lower, torch.eye(len(gram), dtype=gram.dtype), upper=False)
        return transform / scales[None, :], condition
    else:
        # Symmetric Lowdin uses the original source Gram G^(-1/2).
        # Diagonal equilibration preserves the ordered Cholesky convention,
        # but whitening the equilibrated Gram changes the Lowdin coordinates.
        # Independent implementation of the polar-factor formula in Higham,
        # "What Is the Polar Decomposition?" (2020):
        # https://nhigham.com/2020/07/28/what-is-the-polar-decomposition/
        values, vectors = torch.linalg.eigh(gram)
        if float(values.min()) <= 0:
            raise ArithmeticError("Shell-filtered source functions are not linearly independent "
                                  "(nonpositive Gram eigenvalue " + repr(float(values.min())) + ")")
        transform = (vectors * values.rsqrt()) @ vectors.T
        return transform, condition


def _transform(plan, gram, orthogonalization):
    if orthogonalization["scope"] == "joint_source":
        return _block_transform(gram, orthogonalization["factorization"])
    count = int(plan["radial_count"])
    transform = torch.zeros_like(gram)
    worst = 1.
    for shell in range(len(plan["shells"]["centers"])):
        block = slice(shell * count, (shell + 1) * count)
        transform[block, block], condition = _block_transform(gram[block, block], orthogonalization["factorization"])
        worst = max(worst, condition)
    return transform, worst


def _residuals(plan, transform, gram, orthogonalization):
    identity = transform @ gram @ transform.T
    count = int(plan["radial_count"])
    eye = torch.eye(len(gram), dtype=gram.dtype)
    if orthogonalization["scope"] == "joint_source":
        return {"identity_residual": float(torch.linalg.matrix_norm(identity - eye, ord=2)),
                "cross_shell_overlap": 0.}
    blocks = torch.block_diag(*(identity[shell * count:(shell + 1) * count, shell * count:(shell + 1) * count]
                                for shell in range(len(plan["shells"]["centers"]))))
    return {"identity_residual": float((blocks - eye).abs().max()),
            "cross_shell_overlap": float(torch.linalg.matrix_norm(identity - blocks, ord=2))}


def shell_filtered_source_plan(family, radial_count, lmax, shells, orthogonalization=None):
    """Compile the frozen transform of one shell-filtered family.

    ``family`` is a configured name from ``SHELL_FILTERED_BASE_FAMILIES``;
    ``radial_count`` is the number of base radial functions per shell and
    ``lmax`` the largest angular degree whose transform is needed.  The plan
    stores the construction Gram, the transform, the independent-rule
    residuals and a content hash; its ``family_id`` carries that hash.
    """
    name = shell_filtered_family_name(family)
    shells, orthogonalization = normalize_shell_filter_settings(shells, orthogonalization)
    radial_count, lmax = int(radial_count), int(lmax)
    if radial_count < 1 or lmax < 0:
        raise ValueError("radial_count must be positive and lmax nonnegative")
    plan = {"schema": SHELL_FILTERED_SOURCE_PLAN_SCHEMA, "family": name,
            "base_family": SHELL_FILTERED_BASE_FAMILIES[name], "radial_count": radial_count, "lmax": lmax,
            "shell_count": len(shells["centers"]), "shells": {**shells, "centers": list(shells["centers"])},
            "orthogonalization": orthogonalization,
            "quadrature": {"rule": "gauss_legendre_on_unit_interval",
                           "construction_points": orthogonalization["construction_points"],
                           "check_points": orthogonalization["check_points"]},
            "channel_order": "shell_major_radial_minor", "radial_channel_index": "q = shell * radial_count + n",
            "coordinate": "x = r / r_cut_pair", "measure": "x^(2l) x^2 dx on [0, 1]",
            "radial_definition": "T_l[(s,n),(s',n')] w_s'(x) (1-x)^p P_n'(x); x^l in the regular solid harmonic",
            "transforms": {}, "gram": {}, "diagnostics": {}}
    with torch.no_grad():
        for angular_l in range(lmax + 1):
            gram = _gram(plan, angular_l, orthogonalization["construction_points"])
            transform, condition = _transform(plan, gram, orthogonalization)
            construction = _residuals(plan, transform, gram, orthogonalization)
            check = _residuals(plan, transform, _gram(plan, angular_l, orthogonalization["check_points"]),
                               orthogonalization)
            if max(construction["identity_residual"], check["identity_residual"]) > orthogonalization["tolerance"]:
                raise ArithmeticError("Shell-filtered transform does not reach the requested orthogonality "
                                      "tolerance at l=" + str(angular_l) + ": " + repr(check))
            key = str(angular_l)
            plan["transforms"][key] = transform.tolist()
            plan["gram"][key] = gram.tolist()
            plan["diagnostics"][key] = {"equilibrated_condition_number": condition,
                "identity_residual_construction": construction["identity_residual"],
                "identity_residual_check": check["identity_residual"],
                "cross_shell_overlap_check": check["cross_shell_overlap"]}
    plan["plan_hash"] = _stable_hash(plan)
    plan["family_id"] = name + ":" + plan["plan_hash"][:16]
    return plan


def validate_shell_filtered_source_plan(plan, *, recompute=False):
    """Check plan integrity; ``recompute`` re-integrates the independent Gram."""
    required = {"schema", "family", "base_family", "radial_count", "lmax", "shell_count", "shells",
                "orthogonalization", "quadrature", "channel_order", "radial_channel_index", "coordinate",
                "measure", "radial_definition", "transforms", "gram", "diagnostics", "plan_hash", "family_id"}
    if not isinstance(plan, dict) or set(plan) != required or plan["schema"] != SHELL_FILTERED_SOURCE_PLAN_SCHEMA:
        raise ValueError("Unknown shell-filtered radial source plan payload")
    body = {key: value for key, value in plan.items() if key not in {"plan_hash", "family_id"}}
    if _stable_hash(body) != plan["plan_hash"] or plan["family_id"] != plan["family"] + ":" + plan["plan_hash"][:16]:
        raise ValueError("Shell-filtered radial source plan hash mismatch")
    shells, orthogonalization = normalize_shell_filter_settings(plan["shells"], {
        key: plan["orthogonalization"][key] for key in ("scope", "factorization", "measure", "tolerance",
                                                        "construction_points", "check_points")})
    # Plans travel through JSON (catalogue requests, caches, artifacts), so
    # tuples and lists of the same settings are the same plan.
    canonical = json.loads(json.dumps({"shells": shells, "orthogonalization": orthogonalization}))
    stored = json.loads(json.dumps({"shells": plan["shells"], "orthogonalization": plan["orthogonalization"]}))
    if (canonical != stored
            or plan["shell_count"] != len(shells["centers"])
            or plan["base_family"] != SHELL_FILTERED_BASE_FAMILIES[shell_filtered_family_name(plan["family"])]):
        raise ValueError("Shell-filtered radial source plan settings are not canonical")
    width = plan["shell_count"] * int(plan["radial_count"])
    if (set(plan["transforms"]) != {str(l) for l in range(int(plan["lmax"]) + 1)}
            or set(plan["gram"]) != set(plan["transforms"])
            or any(len(rows) != width or any(len(row) != width for row in rows)
                   for table in (plan["transforms"], plan["gram"]) for rows in table.values())):
        raise ValueError("Shell-filtered radial source plan tables do not match the channel layout")
    report = {"passed": True, "family_id": plan["family_id"], "recomputed": bool(recompute), "residuals": {}}
    if recompute:
        with torch.no_grad():
            for key, rows in plan["transforms"].items():
                transform = torch.tensor(rows, dtype=torch.float64)
                residual = _residuals(plan, transform, _gram(plan, int(key), plan["quadrature"]["check_points"]),
                                      plan["orthogonalization"])
                if residual["identity_residual"] > plan["orthogonalization"]["tolerance"]:
                    raise ArithmeticError("Stored shell-filtered transform fails the independent Gram check at l=" + key)
                report["residuals"][key] = residual
    return report


def shell_filtered_source_metadata(plan):
    """Source smoothness/provenance record for graph bindings and reports."""
    return {"radial_family": plan["family_id"], "base_family": plan["base_family"],
            "radial_definition": plan["radial_definition"], "coordinate": plan["coordinate"],
            "normalization": "orthonormal_under_" + plan["measure"].replace(" ", "_") + "_per_l_"
                             + plan["orthogonalization"]["convention"] + "_" + plan["orthogonalization"]["scope"],
            "zero_extension": _ZERO_EXTENSION[plan["base_family"]],
            "cutoff_second_derivative": ("continuous_zero" if plan["base_family"] == CHEBYSHEV_C2_SOURCE_FAMILY
                                         else "one_sided"),
            "shells": dict(plan["shells"]), "shell_count": plan["shell_count"],
            "radial_count_per_shell": plan["radial_count"], "plan_hash": plan["plan_hash"],
            "product_algebra": "none_shell_windows_are_not_polynomial_closed",
            "maturity": "experimental"}
