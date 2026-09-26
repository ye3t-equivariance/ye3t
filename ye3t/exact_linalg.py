"""Small exact integer/rational linear algebra kernels."""

from fractions import Fraction

from ye3t.exact_scalars import ExactRadical, exact_scalar


def _as_fraction(value):
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def exact_matrix_zeros(rows, cols):
    return tuple(tuple(0 for _ in range(int(cols))) for _ in range(int(rows)))


def exact_matrix_from_entries(rows, cols, entries):
    rows = int(rows)
    cols = int(cols)
    data = [[0 for _ in range(cols)] for _ in range(rows)]
    for key, value in dict(entries).items():
        row, col = key
        data[int(row)][int(col)] = value
    return tuple(tuple(row) for row in data)


def exact_matrix_matmul(left, right):
    left = tuple(tuple(row) for row in left)
    right = tuple(tuple(row) for row in right)
    if not left:
        return tuple()
    if not right:
        return tuple(tuple() for _ in range(len(left)))
    inner = len(left[0])
    if any(len(row) != inner for row in left):
        raise ValueError("left matrix rows must have consistent length")
    if len(right) != inner:
        raise ValueError("matrix shapes do not align")
    cols = len(right[0]) if right else 0
    if any(len(row) != cols for row in right):
        raise ValueError("right matrix rows must have consistent length")
    out = []
    for left_row in left:
        out_row = []
        for col in range(cols):
            total = 0
            for index, value in enumerate(left_row):
                total += value * right[index][col]
            out_row.append(total)
        out.append(tuple(out_row))
    return tuple(out)


def exact_matrix_transpose(matrix):
    matrix = tuple(tuple(row) for row in matrix)
    if not matrix:
        return tuple()
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("matrix rows must have consistent length")
    return tuple(tuple(row[col] for row in matrix) for col in range(cols))


def exact_matrix_equal(left, right):
    return tuple(tuple(row) for row in left) == tuple(tuple(row) for row in right)


def exact_matrix_flatten(matrix):
    return tuple(value for row in matrix for value in row)


def exact_rational_rank(rows):
    matrix = [
        [_as_fraction(value) for value in row]
        for row in rows
    ]
    if not matrix:
        return 0
    col_count = len(matrix[0])
    if any(len(row) != col_count for row in matrix):
        raise ValueError("all rows must have the same length")
    rank = 0
    row_count = len(matrix)
    for col in range(col_count):
        pivot = None
        for row in range(rank, row_count):
            if matrix[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][col]
        matrix[rank] = [value / pivot_value for value in matrix[rank]]
        for row in range(row_count):
            if row == rank:
                continue
            factor = matrix[row][col]
            if factor == 0:
                continue
            matrix[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(matrix[row], matrix[rank])
            ]
        rank += 1
        if rank == row_count:
            break
    return int(rank)


def _as_exact_radical(value):
    if isinstance(value, ExactRadical):
        return value
    return exact_scalar(value)


def exact_algebraic_rank_or_none(rows):
    """Return exact rank over supported radical scalars, or ``None`` if unsupported."""

    try:
        matrix = [
            [_as_exact_radical(value) for value in row]
            for row in rows
        ]
    except (TypeError, ValueError):
        return None
    if not matrix:
        return 0
    col_count = len(matrix[0])
    if any(len(row) != col_count for row in matrix):
        raise ValueError("all rows must have the same length")
    rank = 0
    row_count = len(matrix)
    for col in range(col_count):
        pivot = None
        for row in range(rank, row_count):
            if matrix[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        pivot_value = matrix[rank][col]
        try:
            matrix[rank] = [value / pivot_value for value in matrix[rank]]
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        for row in range(row_count):
            if row == rank:
                continue
            factor = matrix[row][col]
            if factor == 0:
                continue
            matrix[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(matrix[row], matrix[rank])
            ]
        rank += 1
        if rank == row_count:
            break
    return int(rank)


__all__ = [
    "exact_algebraic_rank_or_none",
    "exact_matrix_equal",
    "exact_matrix_flatten",
    "exact_matrix_from_entries",
    "exact_matrix_matmul",
    "exact_matrix_transpose",
    "exact_matrix_zeros",
    "exact_rational_rank",
]
