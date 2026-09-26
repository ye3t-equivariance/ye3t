"""Deterministic symbolic Young-partition template expansion."""

import ast
import itertools
import re
from fractions import Fraction


_IMPLICIT_N_PRODUCT = re.compile(r"(?<=\d)N")

_PARTITION_FAMILY_REQUEST_KEYS = frozenset(
    {
        "family_id",
        "mode",
        "templates",
        "explicit_by_rank",
        "part",
        "ranks",
        "rank_min",
        "rank_max",
        "rounding_tolerance",
        "description",
    }
)


def _fraction_text(value):
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return str(value.numerator) + "/" + str(value.denominator)


def _evaluate_node(node, rank):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("partition expressions accept integer constants only")
        return Fraction(int(node.value), 1)
    if isinstance(node, ast.Name):
        if node.id != "N":
            raise ValueError("partition expressions accept only the symbol N")
        return Fraction(int(rank), 1)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, rank)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("unsupported unary operator in partition expression")
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, rank)
        right = _evaluate_node(node.right, rank)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero in partition expression")
            return left / right
        raise ValueError("unsupported binary operator in partition expression")
    raise ValueError("unsupported syntax in partition expression")


def evaluate_partition_expression(expression, rank):
    """Evaluate one rational expression in ``N`` exactly."""

    rank = int(rank)
    if rank <= 0:
        raise ValueError("rank must be positive")
    text = str(expression).replace(" ", "")
    if not text:
        raise ValueError("partition expressions must not be empty")
    text = _IMPLICIT_N_PRODUCT.sub("*N", text)
    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid partition expression " + repr(expression)) from exc
    return _evaluate_node(parsed.body, rank)


def _integer_choices(target, tolerance):
    lower = max(1, (target - tolerance).__ceil__())
    upper = (target + tolerance).__floor__()
    return tuple(range(int(lower), int(upper) + 1))


def apportion_partition_template(rank, expressions, rounding_tolerance=0.5):
    """Map rational target parts to one positive integer partition of ``rank``."""

    rank = int(rank)
    expressions = tuple(str(value) for value in tuple(expressions))
    if rank <= 0:
        raise ValueError("rank must be positive")
    if not expressions:
        raise ValueError("a partition template must contain at least one part")
    tolerance = Fraction(str(rounding_tolerance))
    if tolerance < 0:
        raise ValueError("rounding_tolerance must be nonnegative")
    targets = tuple(evaluate_partition_expression(value, rank) for value in expressions)
    if any(value <= 0 for value in targets):
        return {
            "status": "inapplicable",
            "reason": "template produced a nonpositive target part",
            "rank": rank,
            "expressions": list(expressions),
            "raw_targets": [_fraction_text(value) for value in targets],
            "partition": None,
            "deviations": None,
        }
    choices = tuple(_integer_choices(value, tolerance) for value in targets)
    if any(not values for values in choices):
        return {
            "status": "inapplicable",
            "reason": "no positive integer part is within the rounding tolerance",
            "rank": rank,
            "expressions": list(expressions),
            "raw_targets": [_fraction_text(value) for value in targets],
            "partition": None,
            "deviations": None,
        }
    candidates = []
    for values in itertools.product(*choices):
        if sum(values) != rank:
            continue
        deviations = tuple(abs(Fraction(value, 1) - target) for value, target in zip(values, targets))
        key = (
            max(deviations, default=Fraction(0, 1)),
            sum(deviations, Fraction(0, 1)),
            tuple(-int(value) for value in values),
        )
        candidates.append((key, tuple(int(value) for value in values), deviations))
    if not candidates:
        return {
            "status": "inapplicable",
            "reason": "integer parts within tolerance cannot sum exactly to rank N",
            "rank": rank,
            "expressions": list(expressions),
            "raw_targets": [_fraction_text(value) for value in targets],
            "partition": None,
            "deviations": None,
        }
    _key, allocated, deviations = min(candidates, key=lambda item: item[0])
    partition = tuple(sorted(allocated, reverse=True))
    return {
        "status": "selected",
        "reason": "deterministic constrained apportionment",
        "rank": rank,
        "expressions": list(expressions),
        "raw_targets": [_fraction_text(value) for value in targets],
        "allocated_parts_before_sort": list(allocated),
        "partition": list(partition),
        "deviations": [_fraction_text(value) for value in deviations],
        "maximum_deviation": _fraction_text(max(deviations, default=Fraction(0, 1))),
    }


def expand_partition_templates(ranks, templates, rounding_tolerance=0.5):
    """Expand, canonicalize, and deduplicate symbolic templates by rank."""

    ranks = tuple(dict.fromkeys(int(value) for value in tuple(ranks)))
    if not ranks or any(value <= 0 for value in ranks):
        raise ValueError("ranks must contain positive integers")
    templates = tuple(tuple(str(part) for part in tuple(template)) for template in tuple(templates))
    if not templates or any(not template for template in templates):
        raise ValueError("templates must contain nonempty partition templates")
    records = []
    partitions_by_rank = {}
    for rank in ranks:
        seen = {}
        selected = []
        for template_index, template in enumerate(templates):
            record = apportion_partition_template(
                rank,
                template,
                rounding_tolerance=rounding_tolerance,
            )
            record["template_id"] = "template_" + str(template_index).zfill(3)
            partition = record.get("partition")
            if partition is not None:
                key = tuple(int(value) for value in partition)
                if key in seen:
                    record["status"] = "duplicate"
                    record["reason"] = "same canonical partition as " + seen[key]
                    record["duplicate_of"] = seen[key]
                else:
                    seen[key] = record["template_id"]
                    selected.append(list(key))
            records.append(record)
        partitions_by_rank[str(rank)] = selected
    return {
        "schema": "ye3t_symbolic_partition_templates_v1",
        "rounding_tolerance": _fraction_text(Fraction(str(rounding_tolerance))),
        "ranks": list(ranks),
        "templates": [list(template) for template in templates],
        "partitions_by_rank": partitions_by_rank,
        "records": records,
    }


def _canonical_partition(partition, rank, field_name):
    partition = tuple(int(value) for value in tuple(partition))
    if (
        not partition
        or any(value <= 0 for value in partition)
        or tuple(sorted(partition, reverse=True)) != partition
        or sum(partition) != int(rank)
    ):
        raise ValueError(
            str(field_name)
            + " must be a canonical positive partition of rank "
            + str(int(rank))
        )
    return partition


def _family_applies_to_rank(request, rank):
    configured_ranks = tuple(int(value) for value in request.get("ranks", ()))
    if configured_ranks:
        return int(rank) in configured_ranks
    rank_min = int(request.get("rank_min", 1))
    rank_max = request.get("rank_max")
    return int(rank) >= rank_min and (
        rank_max is None or int(rank) <= int(rank_max)
    )


def _normalized_partition_family_request(
    request,
    family_order,
    default_rounding_tolerance,
):
    request = dict(request)
    unknown = tuple(
        sorted(set(request).difference(_PARTITION_FAMILY_REQUEST_KEYS))
    )
    if unknown:
        raise ValueError(
            "partition family request has unknown fields " + repr(unknown)
        )
    family_id = str(request.get("family_id", "")).strip()
    if not family_id:
        raise ValueError("partition family requests require a nonempty family_id")
    mode = str(request.get("mode", "templates"))
    if mode not in {"templates", "explicit_by_rank", "repeated_part", "sign"}:
        raise ValueError(
            "partition family request mode must be templates, "
            "explicit_by_rank, repeated_part, or sign"
        )
    ranks = tuple(sorted({int(value) for value in request.get("ranks", ())}))
    rank_min = request.get("rank_min")
    rank_max = request.get("rank_max")
    if ranks and (rank_min is not None or rank_max is not None):
        raise ValueError(
            "partition family requests must use ranks or rank_min/rank_max, "
            "not both"
        )
    if any(value <= 0 for value in ranks):
        raise ValueError("partition family request ranks must be positive")
    if not ranks:
        rank_min = 1 if rank_min is None else int(rank_min)
        rank_max = None if rank_max is None else int(rank_max)
        if rank_min <= 0 or (rank_max is not None and rank_max < rank_min):
            raise ValueError("partition family request has an invalid rank interval")
    normalized = {
        "family_id": family_id,
        "family_order": int(family_order),
        "mode": mode,
        "description": str(request.get("description", "")),
        "ranks": list(ranks),
        "rank_min": None if ranks else rank_min,
        "rank_max": None if ranks else rank_max,
        "rounding_tolerance": float(
            request.get(
                "rounding_tolerance", default_rounding_tolerance
            )
        ),
    }
    if normalized["rounding_tolerance"] < 0.0:
        raise ValueError("partition family rounding_tolerance must be nonnegative")
    if mode == "templates":
        templates = tuple(
            tuple(str(part) for part in tuple(template))
            for template in tuple(request.get("templates", ()))
        )
        if not templates or any(not template for template in templates):
            raise ValueError(
                "templates partition families require nonempty templates"
            )
        normalized["templates"] = [list(template) for template in templates]
    elif mode == "explicit_by_rank":
        explicit = {}
        for raw_rank, partitions in dict(
            request.get("explicit_by_rank", {})
        ).items():
            explicit_rank = int(raw_rank)
            if explicit_rank <= 0:
                raise ValueError(
                    "explicit partition family ranks must be positive"
                )
            explicit[str(explicit_rank)] = [
                list(
                    _canonical_partition(
                        partition,
                        explicit_rank,
                        "explicit_by_rank[" + str(explicit_rank) + "]",
                    )
                )
                for partition in tuple(partitions)
            ]
        if not explicit:
            raise ValueError(
                "explicit_by_rank partition families require explicit_by_rank"
            )
        normalized["explicit_by_rank"] = explicit
    elif mode == "repeated_part":
        part = int(request.get("part", 0))
        if part <= 0:
            raise ValueError("repeated_part partition families require part > 0")
        normalized["part"] = part
    return normalized


def expand_partition_family_requests(
    ranks,
    family_requests,
    rounding_tolerance=0.5,
):
    """Lower named symbolic Young families to exact rank partitions.

    ``repeated_part`` describes Young-diagram row lengths, so ``part=2``
    yields ``(2,2,...,2)`` when the rank is divisible by two.  It does not
    describe the unrelated cycle type of a permutation conjugacy class.

    Families are processed in the requested order.  When two families lower
    to the same partition at one rank, the first family owns that exact
    partition and the later record is reported as a duplicate.  Malformed
    requests raise before any partial report can be used by an application.
    """

    ranks = tuple(dict.fromkeys(int(value) for value in tuple(ranks)))
    if not ranks or any(value <= 0 for value in ranks):
        raise ValueError("ranks must contain positive integers")
    default_tolerance = float(rounding_tolerance)
    if default_tolerance < 0.0:
        raise ValueError("rounding_tolerance must be nonnegative")
    requests = tuple(
        _normalized_partition_family_request(
            request,
            index,
            default_tolerance,
        )
        for index, request in enumerate(tuple(family_requests))
    )
    if not requests:
        raise ValueError("family_requests must not be empty")
    family_ids = tuple(request["family_id"] for request in requests)
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("partition family_id values must be unique")
    records = []
    assignments = []
    partitions_by_rank = {str(rank): [] for rank in ranks}
    partitions_by_family_by_rank = {
        request["family_id"]: {str(rank): [] for rank in ranks}
        for request in requests
    }
    seen_by_rank = {rank: {} for rank in ranks}

    for request in requests:
        family_id = request["family_id"]
        family_order = int(request["family_order"])
        for rank in ranks:
            if not _family_applies_to_rank(request, rank):
                records.append(
                    {
                        "family_id": family_id,
                        "family_order": family_order,
                        "request_mode": request["mode"],
                        "rank": int(rank),
                        "partition": None,
                        "status": "inapplicable",
                        "reason": "rank is outside the family request scope",
                    }
                )
                continue
            family_records = []
            if request["mode"] == "templates":
                expanded = expand_partition_templates(
                    (rank,),
                    request["templates"],
                    rounding_tolerance=request.get(
                        "rounding_tolerance", default_tolerance
                    ),
                )
                family_records = [dict(record) for record in expanded["records"]]
            elif request["mode"] == "explicit_by_rank":
                partitions = request["explicit_by_rank"].get(str(rank), ())
                if not partitions:
                    family_records = [
                        {
                            "rank": int(rank),
                            "partition": None,
                            "status": "inapplicable",
                            "reason": "no explicit partition was requested at this rank",
                            "template_id": "explicit_none",
                        }
                    ]
                else:
                    family_records = [
                        {
                            "rank": int(rank),
                            "partition": list(
                                _canonical_partition(
                                    partition,
                                    rank,
                                    "explicit_by_rank[" + str(rank) + "]",
                                )
                            ),
                            "status": "selected",
                            "reason": "explicit_by_rank",
                            "template_id": "explicit_" + str(index).zfill(3),
                        }
                        for index, partition in enumerate(partitions)
                    ]
            elif request["mode"] == "repeated_part":
                part = int(request["part"])
                if int(rank) % part:
                    family_records = [
                        {
                            "rank": int(rank),
                            "partition": None,
                            "status": "inapplicable",
                            "reason": "rank is not divisible by repeated row length",
                            "template_id": "repeated_part_" + str(part),
                        }
                    ]
                else:
                    family_records = [
                        {
                            "rank": int(rank),
                            "partition": [part] * (int(rank) // part),
                            "status": "selected",
                            "reason": "exact repeated Young-diagram row length",
                            "template_id": "repeated_part_" + str(part),
                        }
                    ]
            else:
                family_records = [
                    {
                        "rank": int(rank),
                        "partition": [1] * int(rank),
                        "status": "selected",
                        "reason": "exact sign partition",
                        "template_id": "sign",
                    }
                ]

            for raw_record in family_records:
                record = dict(raw_record)
                record.update(
                    {
                        "family_id": family_id,
                        "family_order": family_order,
                        "request_mode": request["mode"],
                    }
                )
                record["template_id"] = (
                    family_id + ":" + str(record.get("template_id", "request"))
                )
                partition = record.get("partition")
                if partition is not None:
                    partition = _canonical_partition(
                        partition,
                        rank,
                        "lowered partition family " + family_id,
                    )
                    record["partition"] = list(partition)
                    if record.get("status") == "selected":
                        prior = seen_by_rank[rank].get(partition)
                        if prior is not None:
                            record["status"] = "duplicate"
                            record["reason"] = (
                                "same canonical partition as "
                                + str(prior["family_id"])
                                + " at rank "
                                + str(rank)
                            )
                            record["duplicate_of_family_id"] = prior["family_id"]
                            record["duplicate_of_template_id"] = prior[
                                "template_id"
                            ]
                        else:
                            assignment = {
                                "rank": int(rank),
                                "partition": list(partition),
                                "family_id": family_id,
                                "family_order": family_order,
                                "template_id": record["template_id"],
                            }
                            seen_by_rank[rank][partition] = assignment
                            assignments.append(assignment)
                            partitions_by_rank[str(rank)].append(list(partition))
                            partitions_by_family_by_rank[family_id][
                                str(rank)
                            ].append(list(partition))
                records.append(record)

    status_counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("selected", "duplicate", "inapplicable", "invalid")
    }
    return {
        "schema": "ye3t_partition_family_requests_v1",
        "ranks": list(ranks),
        "requested_family_ids": list(family_ids),
        "family_requests": [dict(request) for request in requests],
        "partitions_by_rank": partitions_by_rank,
        "partitions_by_family_by_rank": partitions_by_family_by_rank,
        "partition_family_assignments": assignments,
        "records": records,
        "status_counts": status_counts,
        "invalid_request_policy": "raise_before_partial_lowering",
    }
