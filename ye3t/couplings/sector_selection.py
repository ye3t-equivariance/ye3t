"""Representation limits applied to independent compiler inventories.

Selectors act on entire irreducible multiplets. The source adapter supplies
the actual group degree and independent inventory through its canonical plan.
"""

from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral

from .partition_templates import apportion_partition_template


def normalize_sector_selection(selection):
    """Return a portable selection without discovering or compiling paths."""
    if not isinstance(selection, Mapping):
        raise TypeError("A sector selection must be a mapping")
    unknown = set(selection) - {"young_lambda", "L", "L_max", "parity", "copies"}
    if unknown:
        raise ValueError("Unsupported sector selection fields: " + str(sorted(unknown)))
    result = {}
    young = selection.get("young_lambda", "all")
    if isinstance(young, str):
        young = {"(N)": "symmetric_only", "(1)^N": "antisymmetric_only"}.get(young, young)
        if young not in {"all", "symmetric_only", "antisymmetric_only"}:
            young = [young]
    if not isinstance(young, str):
        if not isinstance(young, (list, tuple)) or not young:
            raise ValueError("young_lambda needs a policy or a nonempty list of partition templates")
        young = list(dict.fromkeys(str(value).replace(" ", "") for value in young))
        for value in young:
            if value == "(1)^N":
                continue
            if not value.startswith("(") or not value.endswith(")"):
                raise ValueError("Young templates use parentheses, for example (N-2,2)")
            # Parse syntax even when the particular family is unavailable at N=1.
            apportion_partition_template(1, value[1:-1].split(","), rounding_tolerance=0)
    result["young_lambda"] = young
    if "L" in selection and "L_max" in selection:
        raise ValueError("Choose either L or L_max in a sector selection")
    for name in ("L", "L_max"):
        if name not in selection:
            continue
        value = selection[name]
        values = value if name == "L" and isinstance(value, (tuple, list)) else [value]
        if not values or any(isinstance(item, bool) or not isinstance(item, Integral)
                             or item < 0 for item in values):
            raise ValueError(name + " must contain nonnegative integer angular momenta")
        result[name] = sorted(set(int(item) for item in values)) if name == "L" else int(value)
    parity = selection.get("parity", "all")
    if parity not in {"all", "even", "odd"}:
        raise ValueError("Sector parity must be all, even or odd")
    result["parity"] = parity
    copies = selection.get("copies", "all")
    if copies != "all" and (isinstance(copies, bool) or not isinstance(copies, Integral) or copies < 1):
        raise ValueError("copies must be all or a positive independent-copy budget per content and sector")
    result["copies"] = copies if copies == "all" else int(copies)
    return result


def resolve_sector_partitions(selection, degree):
    """Resolve Young types at the degree of the acting group, including S_0."""
    if isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 0:
        raise ValueError("The action degree must be a nonnegative integer")
    young = selection["young_lambda"]
    if young == "all":
        return None
    if young == "symmetric_only":
        return ((degree,),) if degree else ((),)
    if young == "antisymmetric_only":
        return ((1,) * degree,)
    result = []
    for template in young:
        if template == "(1)^N":
            partition = (1,) * degree
        elif degree == 0:
            partition = () if template == "(N)" else None
        else:
            resolved = apportion_partition_template(
                degree, template[1:-1].split(","), rounding_tolerance=0)
            partition = None if resolved["partition"] is None else tuple(resolved["partition"])
        if partition is not None and partition not in result:
            result.append(partition)
    return tuple(result)


def sector_is_selected(selection, degree, partition, L, parity):
    """Cheap precompilation test for an already valid irreducible type."""
    partitions = resolve_sector_partitions(selection, degree)
    return ((partitions is None or tuple(partition) in partitions)
            and ("L" not in selection or L in selection["L"])
            and ("L_max" not in selection or L <= selection["L_max"])
            and (selection["parity"] == "all"
                 or parity == (1 if selection["parity"] == "even" else -1)))


def _validate_sector_requests(inventory, selection):
    """Check explicit requests against available types without expanding copies."""
    alternatives = sorted({(row["action_degree"], tuple(row["young_lambda"]), row["L"], row["parity"])
                           for row in inventory})
    if "L" in selection:
        unavailable = [L for L in selection["L"] if not any(
            sector_is_selected(dict(selection, L=[L]), row["action_degree"], row["young_lambda"],
                               row["L"], row["parity"]) for row in inventory)]
        if unavailable:
            raise ValueError("Unavailable explicit angular requests: " + str(unavailable)
                             + ". Available (action degree, Young type, L, parity): " + str(alternatives))
    young = selection["young_lambda"]
    if isinstance(young, list):
        unavailable = []
        for template in young:
            single = dict(selection, young_lambda=[template])
            if not any(sector_is_selected(single, row["action_degree"], row["young_lambda"],
                                          row["L"], row["parity"]) for row in inventory):
                unavailable.append(template)
        if unavailable:
            raise ValueError("Unavailable explicit Young requests: " + str(unavailable)
                             + ". Available (action degree, Young type, L, parity): " + str(alternatives))
    return alternatives


def select_independent_sector_copies(inventory, selection):
    """Select independent inventory entries in canonical copy order.

    Each row supplies content_id, action_degree, young_lambda, L, parity and
    coordinate_id. Distinct contents remain separate multiplicity spaces.
    The input inventory must already exclude repeated-content dependencies.
    """
    selection = normalize_sector_selection(selection)
    inventory = list(inventory)
    alternatives = _validate_sector_requests(inventory, selection)
    groups = defaultdict(list)
    seen = set()
    for row in inventory:
        if row["coordinate_id"] in seen:
            raise ValueError("Independent sector inventory repeats a coordinate identity")
        seen.add(row["coordinate_id"])
        key = (row["content_id"], row["action_degree"], tuple(row["young_lambda"]), row["L"], row["parity"])
        groups[key].append(row)
    selected, report = [], []
    for (_, degree, partition, L, parity), rows in groups.items():
        allowed = sector_is_selected(selection, degree, partition, L, parity)
        count = len(rows) if selection["copies"] == "all" else min(len(rows), selection["copies"])
        kept = rows[:count] if allowed else []
        selected.extend(kept)
        report.append({"content_id": rows[0]["content_id"], "action_degree": degree,
                       "young_lambda": list(partition), "L": L, "parity": parity,
                       "available_copies": len(rows), "selected_copies": len(kept)})
    if not selected:
        raise ValueError("No available independent source sector matches the selection. "
                         "Available (action degree, Young type, L, parity): " + str(alternatives))
    return {"selection": selection, "selected": selected, "sector_inventory": report,
            "available_copies": len(inventory), "selected_copies": len(selected),
            "selection_unit": "complete_irreducible_multiplet"}
