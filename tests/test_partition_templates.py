import pytest


def test_symbolic_partition_templates_handle_odd_rank_and_deduplicate():
    from ye3t.couplings import expand_partition_templates

    report = expand_partition_templates(
        (4, 5),
        (
            ("N",),
            ("N-1", "1"),
            ("N/2", "N/2"),
            ("N-2", "2"),
        ),
    )

    assert report["partitions_by_rank"]["4"] == [[4], [3, 1], [2, 2]]
    assert report["partitions_by_rank"]["5"] == [[5], [4, 1], [3, 2]]
    rank5_balanced = next(
        row
        for row in report["records"]
        if row["rank"] == 5 and row["expressions"] == ["N/2", "N/2"]
    )
    assert rank5_balanced["partition"] == [3, 2]
    assert rank5_balanced["allocated_parts_before_sort"] == [3, 2]
    assert rank5_balanced["maximum_deviation"] == "1/2"
    duplicates = [row for row in report["records"] if row["status"] == "duplicate"]
    assert {tuple(row["partition"]) for row in duplicates} == {(2, 2), (3, 2)}


def test_symbolic_partition_templates_report_inapplicable_apportionment():
    from ye3t.couplings import apportion_partition_template

    record = apportion_partition_template(
        4,
        ("2N/3", "N/6", "N/6"),
        rounding_tolerance=0.5,
    )

    assert record["status"] == "inapplicable"
    assert record["partition"] is None
    assert "sum exactly" in record["reason"]


def test_partition_expression_parser_is_exact_and_rejects_code():
    from fractions import Fraction

    from ye3t.couplings import evaluate_partition_expression

    assert evaluate_partition_expression("2N/3", 5) == Fraction(10, 3)
    assert evaluate_partition_expression("N-2", 5) == Fraction(3, 1)
    with pytest.raises(ValueError, match="only the symbol N"):
        evaluate_partition_expression("rank/2", 5)
    with pytest.raises(ValueError, match="unsupported"):
        evaluate_partition_expression("N**2", 5)


def test_named_partition_families_lower_exactly_and_report_duplicates():
    from ye3t.couplings import expand_partition_family_requests

    report = expand_partition_family_requests(
        (4, 6, 8),
        (
            {
                "family_id": "trivial",
                "mode": "templates",
                "templates": (("N",),),
            },
            {
                "family_id": "balanced_two_row",
                "mode": "templates",
                "templates": (("N/2", "N/2"),),
            },
            {
                "family_id": "all_twos",
                "mode": "repeated_part",
                "part": 2,
            },
            {
                "family_id": "proportional_2_1_1",
                "mode": "templates",
                "templates": (("N/2", "N/4", "N/4"),),
                "ranks": (8,),
            },
            {"family_id": "sign", "mode": "sign", "ranks": (8,)},
        ),
    )

    assert report["partitions_by_rank"] == {
        "4": [[4], [2, 2]],
        "6": [[6], [3, 3], [2, 2, 2]],
        "8": [[8], [4, 4], [2, 2, 2, 2], [4, 2, 2], [1] * 8],
    }
    assert report["status_counts"] == {
        "selected": 10,
        "duplicate": 1,
        "inapplicable": 4,
        "invalid": 0,
    }
    duplicate = next(
        record
        for record in report["records"]
        if record["family_id"] == "all_twos" and record["rank"] == 4
    )
    assert duplicate["status"] == "duplicate"
    assert duplicate["duplicate_of_family_id"] == "balanced_two_row"
    assignments = {
        (row["rank"], tuple(row["partition"])): row["family_id"]
        for row in report["partition_family_assignments"]
    }
    assert assignments[(8, (4, 2, 2))] == "proportional_2_1_1"
    assert assignments[(8, (1,) * 8)] == "sign"


def test_named_partition_families_reject_malformed_requests():
    from ye3t.couplings import expand_partition_family_requests

    with pytest.raises(ValueError, match="unique"):
        expand_partition_family_requests(
            (4,),
            (
                {"family_id": "same", "mode": "sign"},
                {"family_id": "same", "mode": "templates", "templates": (("N",),)},
            ),
        )
    with pytest.raises(ValueError, match="canonical positive partition"):
        expand_partition_family_requests(
            (4,),
            (
                {
                    "family_id": "bad",
                    "mode": "explicit_by_rank",
                    "explicit_by_rank": {"4": ((1, 3),)},
                },
            ),
        )
