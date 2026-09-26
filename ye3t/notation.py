"""Concise publication-aligned labels for exact sectors and basis vectors."""
from ye3t._record import recordclass
import re


_BASIS_LABEL_FIELDS = (
    "content",
    "radial_content",
    "block_young",
    "global_young",
    "tableau",
    "block_angular",
    "total_L",
    "M",
    "multiplicity",
    "lr_path",
    "rotational_path",
    "tree_path",
    "parity",
    "normalization",
    "angular_convention",
    "carrier",
)


@recordclass(('content', 'radial_content', 'block_young', 'global_young', 'tableau', 'block_angular', 'total_L', 'M', 'multiplicity', 'lr_path', 'rotational_path', 'tree_path', 'parity', 'normalization', 'angular_convention', 'carrier'), frozen = True)
class BasisLabel:
    """Complete schema for one YE3T basis coordinate.

    Fields that are not known from a legacy compact label remain ``None``.
    Formatters must surface only known values and must not invent missing
    Young, LR, tableau, parity, or convention data.
    """

    content = None
    radial_content = None
    block_young = None
    global_young = None
    tableau = None
    block_angular = None
    total_L = None
    M = None
    multiplicity = None
    lr_path = None
    rotational_path = None
    tree_path = None
    parity = None
    normalization = None
    angular_convention = None
    carrier = None

    def to_dict(self, *, include_none=False):
        data = {field: _json_ready(getattr(self, field)) for field in _BASIS_LABEL_FIELDS}
        if include_none:
            return data
        return {key: value for key, value in data.items() if value is not None}


def _json_ready(value):
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _compact_atom(value):
    if isinstance(value, (tuple, list)):
        return "(" + ",".join(_compact_atom(item) for item in value) + ")"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_compact_atom(value[key])}" for key in sorted(value)) + "}"
    return str(value)


def to_compact(label):
    """Return a stable compact text representation of a ``BasisLabel``."""

    label = coerce_basis_label(label)
    parts = []
    for field in _BASIS_LABEL_FIELDS:
        value = getattr(label, field)
        if value is not None:
            parts.append(f"{field}={_compact_atom(value)}")
    return "BasisLabel[" + ";".join(parts) + "]"


def to_json(label):
    """Return a JSON-serializable dictionary for a ``BasisLabel``."""

    return coerce_basis_label(label).to_dict(include_none=True)


def to_human(label):
    """Return a readable one-line basis label, omitting unknown components."""

    label = coerce_basis_label(label)
    data = label.to_dict(include_none=False)
    if not data:
        return "BasisLabel[unknown]"
    return "BasisLabel[" + "; ".join(f"{key}={_compact_atom(value)}" for key, value in data.items()) + "]"


def to_latex(label):
    """Return a compact LaTeX-ish representation of known basis coordinates."""

    label = coerce_basis_label(label)
    pieces = []
    if label.content is not None:
        pieces.append(r"\nu=" + _compact_atom(label.content))
    if label.radial_content is not None:
        pieces.append(r"\kappa=" + _compact_atom(label.radial_content))
    if label.block_young is not None:
        pieces.append(r"\boldsymbol{\mu}=" + _compact_atom(label.block_young))
    if label.global_young is not None:
        pieces.append(r"\lambda=" + _compact_atom(label.global_young))
    if label.block_angular is not None:
        pieces.append(r"\boldsymbol{\Lambda}=" + _compact_atom(label.block_angular))
    if label.total_L is not None:
        pieces.append("L=" + _compact_atom(label.total_L))
    if label.M is not None:
        pieces.append("M=" + _compact_atom(label.M))
    if label.multiplicity is not None:
        pieces.append(r"\alpha=" + _compact_atom(label.multiplicity))
    if label.lr_path is not None:
        pieces.append(r"\gamma=" + _compact_atom(label.lr_path))
    if label.rotational_path is not None:
        pieces.append(r"\pi=" + _compact_atom(label.rotational_path))
    if label.tree_path is not None:
        pieces.append(r"\Gamma_T=" + _compact_atom(label.tree_path))
    if label.parity is not None:
        pieces.append(r"\epsilon=" + _compact_atom(label.parity))
    if label.carrier is not None:
        pieces.append(r"\mathrm{carrier}=" + _compact_atom(label.carrier))
    return "$" + r", ".join(pieces) + "$"


def to_filename_safe(label):
    """Return a filename-safe basis-label string."""

    text = to_compact(label)
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text)
    return text.strip("_")


def coerce_basis_label(label=None, *, index=None, quotient="full", carrier="ACE_density"):
    """Coerce supported legacy label objects into ``BasisLabel``."""

    if isinstance(label, BasisLabel):
        return label
    if hasattr(label, "handle") and hasattr(label, "compact_label"):
        return basis_label_from_entry(label, quotient=quotient, carrier=carrier)
    if label is None:
        return BasisLabel(multiplicity=index, normalization=quotient, carrier=carrier)
    return basis_label_from_compact(label, index=index, quotient=quotient, carrier=carrier)


def basis_label_from_compact(label, *, index=None, quotient="full", carrier="ACE_density"):
    """Build a schema label from a legacy compact exact ACE label."""

    from ye3t.core.labels import normalize_compact_label

    compact = normalize_compact_label(label)
    tree_path = compact.tree_type
    if compact.basis_key:
        tree_path = {"tree_type": compact.tree_type, "basis_key": compact.basis_key}
    return BasisLabel(
        content=tuple(compact.n_tuple),
        radial_content=tuple(compact.n_tuple),
        block_angular=tuple(compact.l_tuple),
        total_L=int(compact.L_R),
        multiplicity=index,
        rotational_path=tuple(compact.internal_Ls),
        tree_path=tree_path,
        normalization=quotient,
        angular_convention="compact-exact-ace-v1",
        carrier=carrier,
    )


def basis_label_from_entry(entry, *, quotient="full", carrier="ACE_density"):
    """Build a schema label from an exact basis-sector entry."""

    metadata = getattr(entry, "metadata", None)
    handle = getattr(entry, "handle", None)
    compact = getattr(entry, "compact_label", None)
    if metadata is None:
        return basis_label_from_compact(
            compact,
            index=None if handle is None else int(handle.basis_index),
            quotient=quotient,
            carrier=carrier,
        )
    block_metadata = tuple(getattr(metadata, "young_block_multiplicities", tuple()))
    block_young = tuple((int(block.block_size),) for block in block_metadata) if block_metadata else None
    block_angular = tuple(int(block.Lambda) for block in block_metadata) if block_metadata else tuple(metadata.l_tuple)
    return BasisLabel(
        content=tuple(metadata.eta_tuple),
        radial_content=tuple(metadata.eta_tuple),
        block_young=block_young,
        global_young=(sum(int(x) for x in metadata.eta_tuple),) if carrier == "ACE_density" else None,
        block_angular=block_angular,
        total_L=int(metadata.root_L),
        multiplicity=None if handle is None else int(handle.basis_index),
        rotational_path=tuple(getattr(metadata.reconstruction, "internal_Ls_postorder", tuple())),
        tree_path=getattr(metadata, "canonical_tree_signature", None),
        normalization=quotient,
        angular_convention=getattr(metadata.reconstruction, "phase_convention", None),
        carrier=carrier,
    )


def _group_repeated_pairs(nin, lin):
    groups = []
    index_by_pair = {}
    for idx, pair in enumerate(zip(tuple(nin), tuple(lin))):
        pair = (int(pair[0]), int(pair[1]))
        if pair not in index_by_pair:
            index_by_pair[pair] = len(groups)
            groups.append([pair, []])
        groups[index_by_pair[pair]][1].append(int(idx))
    return tuple((pair, tuple(indices)) for pair, indices in groups)


def format_nu(nin, lin):
    """Return a compact ``boldsymbol nu`` text with repeated leaves grouped."""
    parts = []
    for pair, indices in _group_repeated_pairs(nin, lin):
        eta, ell = pair
        base = f"(eta={eta},l={ell})"
        if len(indices) > 1:
            base = f"{base}^{len(indices)}"
        parts.append(base)
    return "(" + ",".join(parts) + ")"


def format_young_subgroup(nin, lin):
    """Return the repeated-factor subgroup associated with ``boldsymbol nu``."""
    factors = []
    for _, indices in _group_repeated_pairs(nin, lin):
        if len(indices) > 1:
            body = ",".join(str(i) for i in indices)
            factors.append(f"S{len(indices)}{{{body}}}")
    if not factors:
        return "1"
    return " x ".join(factors)


def format_block_trivial_lambda(nin, lin):
    """Return the block-trivial Young character for ordinary ACE sectors."""
    parts = []
    for _, indices in _group_repeated_pairs(nin, lin):
        if len(indices) > 1:
            parts.append("(" + str(len(indices)) + ")")
    if not parts:
        return "lambda_0=()"
    return "lambda_0=(" + ",".join(parts) + ")"


def format_ye3t_sector(
    nin,
    lin,
    L_R,
    *,
    lambda_text = None,
    parity = None,
    quotient = "full",
    alpha = None,
    convention = "canonical-v1",
):
    """Return an unambiguous sector header for a general ``ye3t`` space."""
    if lambda_text is None:
        lambda_text = format_block_trivial_lambda(nin, lin)
    lambda_field = str(lambda_text)
    if not lambda_field.startswith("lambda"):
        lambda_field = f"lambda={lambda_field}"
    parts = [
        f"nu={format_nu(nin, lin)}",
        f"G_nu={format_young_subgroup(nin, lin)}",
        lambda_field,
        f"L_R={int(L_R)}",
    ]
    if parity is not None:
        parts.append(f"parity={parity}")
    parts.append(f"q={quotient}")
    if alpha is not None:
        parts.append(f"alpha={int(alpha)}")
    parts.append(f"convention={convention}")
    return "Sector: " + "; ".join(parts)


def format_ye3t_basis(index, label = None, *, quotient = "full"):
    """Return the coordinate label for one basis vector inside a sector."""
    basis_label = coerce_basis_label(label, index=int(index), quotient=quotient)
    human = to_human(basis_label)
    if human.startswith("BasisLabel[") and human.endswith("]"):
        human = human[len("BasisLabel["):-1]
    return "Basis: " + human
