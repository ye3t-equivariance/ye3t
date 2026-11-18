"""Cached changes of coordinates between already compiled physical images.

The source and target couplings are unchanged. A row of the returned matrix
expresses one source function in the target's independent coordinates.
"""

from ye3t.cache.artifacts import ArtifactCacheMiss, YE3TArtifactStore
from ye3t.exact_scalars import ExactRadical
from .lifted_cauchy_scalar import _exact_scalar_payload


def tagged_source_coordinates(source, target):
    """Return F_source = F_target K.T, with sparse exact rows of K.

Identical coordinates and stored source relations are reused directly. When
necessary, the existing source-image LDL operation resolves only connected
physical polynomial blocks. This preparation is persisted and never executed
during descriptor evaluation or molecular dynamics.
    """
    for value in (source, target):
        if value.payload.get("coordinate_policy") != "exact_physical_pivots_v1":
            raise ValueError("Source-coordinate maps require the general physical-image convention.")
    source_request, target_request = source.plan.report.request, target.plan.report.request
    if (tuple(source_request["species"]) != tuple(target_request["species"]) or
            source_request["catalogue"].get("source_profile", "B") !=
            target_request["catalogue"].get("source_profile", "B")):
        raise ValueError("Coordinate maps require matching species and physical source profiles.")
    source_ids = tuple(row["coordinate_id"] for row in source.payload["image_coordinate_provenance"])
    target_ids = tuple(row["coordinate_id"] for row in target.payload["image_coordinate_provenance"])
    identity = {"source": source.self_hash, "target": target.self_hash}

    def build():
        from .source_image_gram import _ExactSourceImageGram
        from .tagged_cauchy_image import _moment_term_map
        one = _exact_scalar_payload(ExactRadical.rational(1))
        known = {value: [{"feature_index": index, "coefficient": one}]
                 for index, value in enumerate(target_ids)}
        inventory = target.payload.get("independent_inventory") or {}
        for row in inventory.get("source_reconstruction", ()):
            known.setdefault(row["coordinate_id"], row["image_coordinates"])
        maps = [known.get(value) for value in source_ids]
        missing = [index for index, row in enumerate(maps) if row is None]
        if missing:
            targets = tuple(_moment_term_map(row) for row in target.payload["image_rows"])
            incidence = {}
            for index, row in enumerate(targets):
                for monomial in row:
                    incidence.setdefault(monomial, set()).add(index)
            factorizations = {}
            for index in missing:
                row = _moment_term_map(source.payload["image_rows"][index])
                pending, seen, selected = list(row), set(), set()
                while pending:
                    monomial = pending.pop()
                    if monomial in seen:
                        continue
                    seen.add(monomial)
                    for target_index in incidence.get(monomial, ()):
                        if target_index not in selected:
                            selected.add(target_index)
                            pending.extend(targets[target_index])
                indices = tuple(sorted(selected))
                if indices not in factorizations:
                    image = _ExactSourceImageGram()
                    for target_index in indices:
                        if not image.append(targets[target_index]):
                            raise ArithmeticError("Target physical coordinates are not independent.")
                    factorizations[indices] = image
                image = factorizations[indices]
                if image.append(row):
                    raise ValueError("Selected physical function is outside the target catalogue span.")
                coefficients = image.reconstructions.pop()
                image.cells -= len(coefficients)
                maps[index] = [{"feature_index": indices[j], "coefficient": _exact_scalar_payload(value)}
                               for j, value in sorted(coefficients.items())]
        return {"schema": "ye3t_tagged_source_coordinates_v1", **identity,
                "shape": [len(source_ids), len(target_ids)], "rows": maps}

    def validate(value):
        if (value.get("schema") != "ye3t_tagged_source_coordinates_v1" or
                any(value.get(key) != item for key, item in identity.items()) or
                list(value.get("shape", ())) != [len(source_ids), len(target_ids)] or
                len(value.get("rows", ())) != len(source_ids)):
            raise ValueError("Source-coordinate map identity or dimensions changed.")
        for row in value["rows"]:
            indices = [term["feature_index"] for term in row]
            if len(set(indices)) != len(indices) or any(
                    type(index) is not int or not 0 <= index < len(target_ids) for index in indices):
                raise ValueError("Invalid source-coordinate map indices.")
        return True

    try:
        return YE3TArtifactStore().resolve(
            "source_coordinates", "tagged_source_coordinates_v1", identity,
            builder=build, validator=validate,
            certificate={"passed": True, "checks": {"exact_source_coordinates": True}},
            required_certificate_checks=("exact_source_coordinates",),
            producer={"revision": 1})["payload"]
    except ArtifactCacheMiss:
        return build()
