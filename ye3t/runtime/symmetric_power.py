"""Symmetric-power runtime kernels for repeated-channel blocks."""

from functools import lru_cache
from math import factorial, sqrt
import os

import torch


_FP32_WIDE_ACCUMULATION_MIN_POWER = 24

from ye3t._record import recordclass
from ye3t.core.basis.homogeneous import (
    AlgebraicSymmetricPowerDecomposer,
    homogeneous_basis_states_by_L_numeric,
    occupancy_expansion_to_m_vectors_numeric,
    thaw_numeric_magnetic_expansion,
)

@recordclass(('input_L', 'output_L', 'pair_left', 'pair_right', 'pair_index', 'pair_out', 'pair_value'), frozen = True)
class SymmetricSquarePairTable:
    """Folded real-basis CG table for ``Sym^2(V_L) -> V_output_L``."""

    @property
    def pair_count(self):
        return int(self.pair_left.numel())


@recordclass(('power', 'input_L', 'output_L', 'multiplicity', 'monomial_indices', 'output_basis', 'output_m', 'monomial_value'), frozen = True)
class SymmetricPowerMonomialTable:
    """Folded real-basis monomial table for one symmetric-power output."""

    @property
    def term_count(self):
        return int(self.monomial_value.numel())

    @property
    def output_dim(self):
        return 2 * int(self.output_L) + 1


@recordclass(('power', 'input_L', 'output_L', 'multiplicity', 'monomial_counts', 'output_offsets', 'monomial_value', 'output_index'), frozen = True)
class SymmetricPowerCountTable:
    """Grouped monomial-count table for symmetric-power evaluation."""

    @property
    def term_count(self):
        return int(self.monomial_value.numel())

    @property
    def input_dim(self):
        return 2 * int(self.input_L) + 1

    @property
    def output_dim(self):
        return 2 * int(self.output_L) + 1


_COUNT_TABLE_CACHE = {}
_COMBINED_COUNT_TABLE_CACHE = {}
_PRODUCT_PLAN_COUNT_TABLE_CACHE = {}


def _use_shared_symmetric_power_table(input, grouped, shared):
    coefficient_count = int(grouped[0].shape[0])
    monomial_count = int(shared[0].shape[0])
    policy = os.environ.get(
        "YE3T_SYMMETRIC_POWER_SHARED_TABLE_POLICY",
        "auto",
    )
    if policy not in {"auto", "force", "off"}:
        raise RuntimeError(
            "YE3T_SYMMETRIC_POWER_SHARED_TABLE_POLICY must be auto, "
            "force, or off"
        )
    if policy == "off":
        return False
    if policy == "force":
        return bool(monomial_count > 0 and coefficient_count > 0)
    if monomial_count == 0 or coefficient_count <= monomial_count:
        return False
    reuse = float(coefficient_count) / float(monomial_count)
    batch = int(input.numel()) // int(input.shape[-1])
    if input.device.type == "cuda":
        if reuse < 8.0:
            return False
        if torch.is_grad_enabled() and input.requires_grad:
            return batch >= 256
        return batch >= 1024
    if input.device.type == "cpu":
        return reuse >= 8.0 or (reuse >= 3.0 and batch >= 8)
    return False


def symmetric_power_product_plan_table_report(
    plan,
    *,
    device="cpu",
    dtype=torch.float64,
    imag_tol=1.0e-12,
):
    """Report exact grouped/shared sparse table shape for one plan."""

    grouped, shared, output_dtype = _symmetric_power_product_plan_count_table(
        plan,
        device=torch.device(device),
        dtype=dtype,
        imag_tol=imag_tol,
    )
    coefficient_count = int(grouped[0].shape[0])
    monomial_count = int(shared[0].shape[0])
    table_bytes = int(
        sum(
            tensor.numel() * tensor.element_size()
            for tensor in tuple(grouped) + tuple(shared)
        )
    )
    powers = tuple(
        sorted({int(entry.power) for entry in plan.entries})
    )
    return {
        "channel_count": int(plan.channel_count),
        "descriptor_count": int(plan.descriptor_count),
        "entry_count": int(len(plan.entries)),
        "powers": powers,
        "maximum_power": max(powers) if powers else None,
        "coefficient_count": coefficient_count,
        "unique_monomial_count": monomial_count,
        "coefficient_reuse": (
            float(coefficient_count) / float(monomial_count)
            if monomial_count
            else 0.0
        ),
        "cached_grouped_and_shared_table_bytes": table_bytes,
        "output_dtype": str(output_dtype),
        "selection_policy": os.environ.get(
            "YE3T_SYMMETRIC_POWER_SHARED_TABLE_POLICY",
            "auto",
        ),
        "runtime_path_discovery": False,
    }


def _shared_symmetric_power_table(
    count_rows,
    output_offsets,
    coefficient_outputs,
    coefficient_values,
    input_width,
    table_device,
):
    """Lower output-grouped rows to unique monomials plus sparse coefficients."""

    unique_rows = []
    unique_index = {}
    coefficient_terms = []
    for row in count_rows:
        key = tuple(int(value) for value in row)
        index = unique_index.get(key)
        if index is None:
            index = len(unique_rows)
            unique_index[key] = index
            unique_rows.append(key)
        coefficient_terms.append(index)
    if unique_rows:
        monomial_counts = torch.tensor(
            unique_rows,
            dtype=torch.long,
            device=table_device,
        )
    else:
        monomial_counts = torch.empty(
            (0, int(input_width)),
            dtype=torch.long,
            device=table_device,
        )
    return (
        monomial_counts,
        torch.tensor(
            output_offsets,
            dtype=torch.long,
            device=table_device,
        ),
        torch.tensor(
            coefficient_terms,
            dtype=torch.long,
            device=table_device,
        ),
        torch.tensor(
            coefficient_outputs,
            dtype=torch.long,
            device=table_device,
        ),
        coefficient_values.to(device=table_device),
    )


def _native_real_cg_entries_cpu(*args):
    from ye3t.runtime.native import _real_cg_entries_cpu

    return _real_cg_entries_cpu(*args)


def _native_real_to_complex(*args, **kwargs):
    from ye3t.runtime.native import real_tesseral_to_complex_multiplet

    return real_tesseral_to_complex_multiplet(*args, **kwargs)


def _native_complex_to_real(*args, **kwargs):
    from ye3t.runtime.native import complex_multiplet_to_real_tesseral

    return complex_multiplet_to_real_tesseral(*args, **kwargs)


def _validate_symmetric_power(power):
    power = int(power)
    if power < 2:
        raise ValueError("symmetric power kernels require power >= 2.")
    if power > 64:
        raise ValueError("direct symmetric power kernels currently support powers 2 through 64.")
    return power


def allowed_symmetric_square_outputs(L):
    """Return SO(3) irreps appearing in ``Sym^2(V_L)``."""
    L = int(L)
    return tuple(range(0, 2 * L + 1, 2))


def allowed_symmetric_power_outputs(power, L):
    """Return SO(3) irreps appearing in ``Sym^power(V_L)``."""
    power = _validate_symmetric_power(power)
    return tuple(sorted(AlgebraicSymmetricPowerDecomposer.decompose(int(L), power)))


def symmetric_power_output_multiplicity(power, L, output_L):
    """Return the multiplicity of ``V_output_L`` in ``Sym^power(V_L)``."""
    power = _validate_symmetric_power(power)
    return int(AlgebraicSymmetricPowerDecomposer.decompose(int(L), power).get(int(output_L), 0))


def _validate_symmetric_square_inputs(x, L, output_L):
    L = int(L)
    output_L = int(output_L)
    if output_L not in allowed_symmetric_square_outputs(L):
        raise ValueError(f"Sym^2(V_{L}) does not contain V_{output_L}.")
    if x.shape[-1] != 2 * L + 1:
        raise ValueError(f"Expected final dimension {2 * L + 1} for L={L}, got {x.shape[-1]}.")
    return L, output_L


def _validate_symmetric_power_inputs(x, power, L, output_L):
    power = _validate_symmetric_power(power)
    L = int(L)
    output_L = int(output_L)
    if output_L != power * L and output_L not in allowed_symmetric_power_outputs(power, L):
        raise ValueError(f"Sym^{power}(V_{L}) does not contain V_{output_L}.")
    if x.shape[-1] != 2 * L + 1:
        raise ValueError(f"Expected final dimension {2 * L + 1} for L={L}, got {x.shape[-1]}.")
    return power, L, output_L


def _validate_optimization_policy(optimization_policy):
    optimization_policy = str(optimization_policy)
    if optimization_policy not in {"off", "auto", "aggressive", "product_evaluator"}:
        raise ValueError(
            "optimization_policy must be one of 'off', 'auto', 'aggressive', or 'product_evaluator'."
        )
    return optimization_policy


@lru_cache(maxsize=None)
def _real_to_complex_matrix_cpu(L):
    L = int(L)
    dim = 2 * L + 1
    basis = torch.eye(dim, dtype=torch.float64)
    converted = _native_real_to_complex(basis, L)
    return tuple(
        tuple(complex(converted[real_idx, magnetic_idx].item()) for real_idx in range(dim))
        for magnetic_idx in range(dim)
    )


@lru_cache(maxsize=None)
def _complex_to_real_matrix_cpu(L):
    L = int(L)
    dim = 2 * L + 1
    rows = [[complex(0.0) for _ in range(dim)] for _ in range(dim)]
    if L == 0:
        rows[0][0] = complex(1.0)
        return tuple(tuple(row) for row in rows)
    rt2 = sqrt(2.0)
    for m_value in range(L, 0, -1):
        real_idx = L - m_value
        rows[-m_value + L][real_idx] = complex(1.0 / rt2)
        rows[m_value + L][real_idx] = complex(((-1) ** m_value) / rt2)
    rows[L][L] = complex(1.0)
    for m_value in range(1, L + 1):
        real_idx = L + m_value
        rows[-m_value + L][real_idx] = complex(0.0, -1.0 / rt2)
        rows[m_value + L][real_idx] = complex(0.0, ((-1) ** m_value) / rt2)
    return tuple(tuple(row) for row in rows)


@lru_cache(maxsize=None)
def _expanded_real_monomials_for_magnetic_tuple(L, magnetic_tuple):
    L = int(L)
    real_to_complex = _real_to_complex_matrix_cpu(L)
    terms = {tuple(): complex(1.0)}
    for m_value in tuple(int(m) for m in magnetic_tuple):
        magnetic_idx = int(m_value) + L
        next_terms = {}
        for prefix, prefix_value in terms.items():
            for real_idx, transform_value in enumerate(real_to_complex[magnetic_idx]):
                if abs(transform_value) <= 1.0e-14:
                    continue
                key = tuple(sorted(prefix + (int(real_idx),)))
                next_terms[key] = next_terms.get(key, 0.0) + prefix_value * transform_value
        terms = next_terms
    return tuple(
        (tuple(key), value)
        for key, value in sorted(terms.items())
        if abs(value) > 1.0e-14
    )


@lru_cache(maxsize=None)
def _expanded_real_monomials_for_occupancy(L, occupancy):
    L = int(L)
    real_to_complex = _real_to_complex_matrix_cpu(L)
    terms = {tuple(): complex(1.0)}
    for magnetic_idx, count in enumerate(tuple(int(x) for x in occupancy)):
        for _ in range(count):
            next_terms = {}
            for prefix, prefix_value in terms.items():
                for real_idx, transform_value in enumerate(real_to_complex[magnetic_idx]):
                    if abs(transform_value) <= 1.0e-14:
                        continue
                    key = tuple(sorted(prefix + (int(real_idx),)))
                    next_terms[key] = next_terms.get(key, 0.0) + prefix_value * transform_value
            terms = next_terms
    return tuple(
        (tuple(key), value)
        for key, value in sorted(terms.items())
        if abs(value) > 1.0e-14
    )


def _lower_stretched_vectors(power, L):
    power = int(power)
    L = int(L)
    vectors = {}
    current_M = int(power) * L
    current = {tuple([0] * (2 * L) + [int(power)]): 1.0}
    vectors[current_M] = dict(current)
    for next_M in range(current_M - 1, -current_M - 1, -1):
        lowered = {}
        for occupancy, coeff in current.items():
            occupancy = tuple(int(x) for x in occupancy)
            for idx in range(1, 2 * L + 1):
                count = occupancy[idx]
                if not count:
                    continue
                m_value = idx - L
                ladder = sqrt((L + m_value) * (L - m_value + 1))
                new_occupancy = list(occupancy)
                new_occupancy[idx] -= 1
                new_occupancy[idx - 1] += 1
                new_occ = tuple(new_occupancy)
                value = coeff * ladder * sqrt(count * (occupancy[idx - 1] + 1))
                lowered[new_occ] = lowered.get(new_occ, 0.0) + value
        total_L = power * L
        denom = sqrt((total_L + current_M) * (total_L - current_M + 1))
        current = {
            occupancy: value / denom
            for occupancy, value in lowered.items()
            if abs(value) > 1.0e-14
        }
        current_M = next_M
        vectors[current_M] = dict(current)
    return vectors


@lru_cache(maxsize=None)
def _symmetric_stretched_monomial_entries_cpu(power, L):
    power = int(power)
    L = int(L)
    output_L = int(power) * L
    complex_to_real = _complex_to_real_matrix_cpu(output_L)
    folded = {}
    for M, vector in _lower_stretched_vectors(power, L).items():
        magnetic_idx = int(M) + output_L
        for occupancy, coeff in vector.items():
            denom = 1
            for count in occupancy:
                denom *= factorial(int(count))
            norm_factor = sqrt(factorial(power) / float(denom))
            base = complex(float(coeff) * norm_factor)
            for monomial_indices, monomial_value in _expanded_real_monomials_for_occupancy(L, occupancy):
                for output_m, output_value in enumerate(complex_to_real[magnetic_idx]):
                    value = base * monomial_value * output_value
                    real_value = float(complex(value).real)
                    if abs(real_value) <= 1.0e-12:
                        continue
                    key = (0, int(output_m), tuple(monomial_indices))
                    folded[key] = folded.get(key, 0.0) + real_value
    return tuple(
        (output_basis, output_m, monomial_indices, float(value))
        for (output_basis, output_m, monomial_indices), value in sorted(folded.items())
        if abs(float(value)) > 1.0e-12
    )


@lru_cache(maxsize=None)
def _symmetric_power_monomial_entries_cpu(power, L, output_L):
    power = _validate_symmetric_power(power)
    L = int(L)
    output_L = int(output_L)
    if output_L == power * L:
        return _symmetric_stretched_monomial_entries_cpu(power, L), 1
    states_by_L = homogeneous_basis_states_by_L_numeric(power, L, basis_mode="orthogonal")
    states = states_by_L.get(output_L, tuple())
    if not states:
        return tuple(), 0
    complex_to_real = _complex_to_real_matrix_cpu(output_L)
    folded = {}
    for output_basis, state in enumerate(states):
        frozen = occupancy_expansion_to_m_vectors_numeric(
            L,
            power,
            state.occupancy_expansion_by_M,
            1.0e-14,
        )
        expansion = thaw_numeric_magnetic_expansion(frozen)
        for M, block in expansion.items():
            magnetic_idx = int(M) + output_L
            for magnetic_tuple, coefficient in block.items():
                base = complex(coefficient)
                for monomial_indices, monomial_value in _expanded_real_monomials_for_magnetic_tuple(
                    L,
                    tuple(magnetic_tuple),
                ):
                    for output_m, output_value in enumerate(complex_to_real[magnetic_idx]):
                        value = base * monomial_value * output_value
                        real_value = float(complex(value).real)
                        if abs(real_value) <= 1.0e-12:
                            continue
                        key = (int(output_basis), int(output_m), tuple(monomial_indices))
                        folded[key] = folded.get(key, 0.0) + real_value
    entries = []
    for (output_basis, output_m, monomial_indices), value in sorted(folded.items()):
        real_value = float(value)
        if abs(real_value) > 1.0e-12:
            entries.append((output_basis, output_m, monomial_indices, real_value))
    return tuple(entries), len(states)


@lru_cache(maxsize=None)
def _symmetric_square_entries_cpu(L, output_L):
    folded = {}
    for idx_left, idx_right, idx_out, value in _native_real_cg_entries_cpu(int(L), int(L), int(output_L)):
        left = min(int(idx_left), int(idx_right))
        right = max(int(idx_left), int(idx_right))
        key = (left, right, int(idx_out))
        folded[key] = folded.get(key, 0.0) + float(value)
    return tuple(
        (left, right, idx_out, value)
        for (left, right, idx_out), value in sorted(folded.items())
        if abs(float(value)) > 1.0e-14
    )


@lru_cache(maxsize=None)
def _symmetric_square_pair_table_cached(L, output_L, device_text, dtype):
    entries = _symmetric_square_entries_cpu(int(L), int(output_L))
    device = torch.device(device_text)
    return SymmetricSquarePairTable(
        input_L=int(L),
        output_L=int(output_L),
        pair_left=torch.tensor([entry[0] for entry in entries], dtype=torch.long, device=device),
        pair_right=torch.tensor([entry[1] for entry in entries], dtype=torch.long, device=device),
        pair_index=torch.tensor(
            [entry[0] for entry in entries] + [entry[1] for entry in entries],
            dtype=torch.long,
            device=device,
        ),
        pair_out=torch.tensor([entry[2] for entry in entries], dtype=torch.long, device=device),
        pair_value=torch.tensor([entry[3] for entry in entries], dtype=dtype, device=device),
    )


def symmetric_square_pair_table(L, output_L, *, device = None, dtype = None):
    """Return a cached device-local folded pair table for a square block."""
    if dtype is None:
        dtype = torch.float64
    table_device = torch.device("cpu") if device is None else torch.device(device)
    if table_device.type == "cuda" and table_device.index is None:
        table_device = torch.device("cuda", torch.cuda.current_device())
    return _symmetric_square_pair_table_cached(
        int(L),
        int(output_L),
        str(table_device),
        dtype,
    )


@lru_cache(maxsize=None)
def _symmetric_square_dense_table_cached(L, output_L, device_text, dtype):
    entries = _symmetric_square_entries_cpu(int(L), int(output_L))
    device = torch.device(device_text)
    table = torch.zeros(
        (len(entries), 2 * int(output_L) + 1),
        dtype=dtype,
        device=device,
    )
    if entries:
        rows = torch.arange(len(entries), dtype=torch.long, device=device)
        outputs = torch.tensor(
            [entry[2] for entry in entries],
            dtype=torch.long,
            device=device,
        )
        values = torch.tensor(
            [entry[3] for entry in entries],
            dtype=dtype,
            device=device,
        )
        table[rows, outputs] = values
    return table


def _symmetric_square_dense_table(L, output_L, x):
    device = x.device
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _symmetric_square_dense_table_cached(
        int(L),
        int(output_L),
        str(device),
        x.dtype,
    )


def symmetric_power_monomial_table(power, L, output_L, *, device = None, dtype = None):
    """Return a device-local monomial table for one symmetric-power output."""
    power, L, output_L = _validate_symmetric_power_inputs(
        torch.empty(2 * int(L) + 1),
        power,
        L,
        output_L,
    )
    entries, multiplicity = _symmetric_power_monomial_entries_cpu(power, L, output_L)
    if dtype is None:
        dtype = torch.float64
    if entries:
        monomial_indices = torch.tensor(
            [entry[2] for entry in entries],
            dtype=torch.long,
            device=device,
        )
    else:
        monomial_indices = torch.empty((0, power), dtype=torch.long, device=device)
    return SymmetricPowerMonomialTable(
        power=power,
        input_L=L,
        output_L=output_L,
        multiplicity=int(multiplicity),
        monomial_indices=monomial_indices,
        output_basis=torch.tensor([entry[0] for entry in entries], dtype=torch.long, device=device),
        output_m=torch.tensor([entry[1] for entry in entries], dtype=torch.long, device=device),
        monomial_value=torch.tensor([entry[3] for entry in entries], dtype=dtype, device=device),
    )


def _symmetric_power_count_table(power, L, output_L, *, device = None, dtype = None):
    """Return a grouped-count table for one symmetric-power output."""
    power, L, output_L = _validate_symmetric_power_inputs(
        torch.empty(2 * int(L) + 1),
        power,
        L,
        output_L,
    )
    entries, multiplicity = _symmetric_power_monomial_entries_cpu(power, L, output_L)
    if dtype is None:
        dtype = torch.float64
    if device is None:
        table_device = torch.device("cpu")
    else:
        table_device = torch.device(device)
        if table_device.type == "cuda" and table_device.index is None:
            table_device = torch.device("cuda", torch.cuda.current_device())
    key = (power, L, output_L, str(table_device), str(dtype))
    cached = _COUNT_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    if entries:
        counts = []
        output_index = []
        for entry in entries:
            row = [0] * (2 * L + 1)
            for index in entry[2]:
                row[int(index)] += 1
            counts.append(tuple(row))
            output_index.append(int(entry[0]) * (2 * int(output_L) + 1) + int(entry[1]))
        monomial_counts = torch.tensor(counts, dtype=torch.long, device=table_device)
        output_index = torch.tensor(output_index, dtype=torch.long, device=table_device)
    else:
        monomial_counts = torch.empty((0, 2 * L + 1), dtype=torch.long, device=table_device)
        output_index = torch.empty((0,), dtype=torch.long, device=table_device)
    output_size = int(multiplicity) * (2 * int(output_L) + 1)
    offsets = [0] * (output_size + 1)
    cursor = 0
    for output_slot in range(output_size):
        while cursor < len(entries):
            entry_index = int(entries[cursor][0]) * (2 * int(output_L) + 1) + int(entries[cursor][1])
            if entry_index != output_slot:
                break
            cursor += 1
        offsets[output_slot + 1] = cursor
    table = SymmetricPowerCountTable(
        power=power,
        input_L=L,
        output_L=output_L,
        multiplicity=int(multiplicity),
        monomial_counts=monomial_counts,
        output_index=output_index,
        output_offsets=torch.tensor(offsets, dtype=torch.long, device=table_device),
        monomial_value=torch.tensor([entry[3] for entry in entries], dtype=dtype, device=table_device),
    )
    _COUNT_TABLE_CACHE[key] = table
    return table


def _combined_symmetric_power_count_table(
    power,
    L,
    output_Ls,
    *,
    device,
    dtype,
):
    """Combine several compiler count tables into one grouped contraction."""

    power = _validate_symmetric_power(power)
    L = int(L)
    output_Ls = tuple(int(value) for value in output_Ls)
    if len(set(output_Ls)) != len(output_Ls):
        raise ValueError("symmetric-power output_Ls must be unique")
    allowed = set(int(value) for value in allowed_symmetric_power_outputs(power, L))
    invalid = tuple(
        value
        for value in output_Ls
        if value not in allowed and value != power * L
    )
    if invalid:
        raise ValueError(
            f"Sym^{power}(V_{L}) does not contain output irreps {invalid}"
        )
    table_device = torch.device(device)
    if table_device.type == "cuda" and table_device.index is None:
        table_device = torch.device("cuda", torch.cuda.current_device())
    key = (
        power,
        L,
        output_Ls,
        str(table_device),
        str(dtype),
    )
    cached = _COMBINED_COUNT_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    count_rows = []
    output_indices = []
    values = []
    offsets = [0]
    output_slices = []
    term_cursor = 0
    output_cursor = 0
    for output_L in output_Ls:
        table = _symmetric_power_count_table(
            power,
            L,
            output_L,
            device="cpu",
            dtype=dtype,
        )
        output_width = int(table.multiplicity) * (2 * output_L + 1)
        count_rows.append(table.monomial_counts)
        output_indices.append(table.output_index + output_cursor)
        values.append(table.monomial_value)
        local_offsets = table.output_offsets.tolist()
        offsets.extend(
            term_cursor + int(value)
            for value in local_offsets[1:]
        )
        output_slices.append(
            (
                output_L,
                int(table.multiplicity),
                output_cursor,
                output_cursor + output_width,
            )
        )
        term_cursor += int(table.term_count)
        output_cursor += output_width

    input_width = 2 * L + 1
    if count_rows:
        monomial_counts = torch.cat(count_rows, dim=0)
        output_index = torch.cat(output_indices, dim=0)
        monomial_value = torch.cat(values, dim=0)
    else:
        monomial_counts = torch.empty(
            (0, input_width),
            dtype=torch.long,
        )
        output_index = torch.empty((0,), dtype=torch.long)
        monomial_value = torch.empty((0,), dtype=dtype)
    grouped = (
        monomial_counts.to(device=table_device),
        torch.tensor(offsets, dtype=torch.long, device=table_device),
        output_index.to(device=table_device),
        monomial_value.to(device=table_device),
    )
    shared = _shared_symmetric_power_table(
        monomial_counts.tolist(),
        offsets,
        output_index.tolist(),
        monomial_value,
        input_width,
        table_device,
    )
    combined = (grouped, shared, tuple(output_slices))
    _COMBINED_COUNT_TABLE_CACHE[key] = combined
    return combined


def _symmetric_power_product_plan_count_table(
    plan,
    *,
    device,
    dtype,
    imag_tol,
):
    """Lower a compiler product plan to one grouped full-channel table."""

    channel_count = int(plan.channel_count)
    descriptor_count = int(plan.descriptor_count)
    table_device = torch.device(device)
    if table_device.type == "cuda" and table_device.index is None:
        table_device = torch.device("cuda", torch.cuda.current_device())
    coefficients_are_complex = any(
        abs(complex(term["coefficient"]).imag) > float(imag_tol)
        for entry in plan.entries
        for term in entry.component_terms
    )
    if dtype in (torch.complex64, torch.complex128):
        output_dtype = dtype
    elif coefficients_are_complex:
        output_dtype = (
            torch.complex128
            if dtype == torch.float64
            else torch.complex64
        )
    else:
        output_dtype = dtype
    key = (
        str(plan.convention_hash),
        channel_count,
        descriptor_count,
        str(table_device),
        str(output_dtype),
        float(imag_tol),
    )
    cached = _PRODUCT_PLAN_COUNT_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    accumulated = {}
    for entry in plan.entries:
        descriptor_index = int(entry.descriptor_index)
        if descriptor_index < 0 or descriptor_index >= descriptor_count:
            raise ValueError(
                "symmetric-power plan descriptor index is outside "
                "descriptor_count"
            )
        channel_indices = tuple(
            int(value) for value in entry.channel_indices
        )
        for term in entry.component_terms:
            local_exponents = tuple(
                int(value) for value in term["exponents"]
            )
            if len(local_exponents) != len(channel_indices):
                raise ValueError(
                    "symmetric-power plan exponent width does not match "
                    "its channel indices"
                )
            counts = [0] * channel_count
            for local_index, exponent in enumerate(local_exponents):
                channel_index = channel_indices[local_index]
                if channel_index < 0 or channel_index >= channel_count:
                    raise ValueError(
                        "symmetric-power plan channel index is outside "
                        "channel_count"
                    )
                counts[channel_index] += int(exponent)
            counts = tuple(counts)
            map_key = (descriptor_index, counts)
            accumulated[map_key] = (
                accumulated.get(map_key, 0.0 + 0.0j)
                + complex(term["coefficient"])
            )

    rows = []
    offsets = [0]
    for descriptor_index in range(descriptor_count):
        descriptor_rows = [
            (counts, coefficient)
            for (row_descriptor, counts), coefficient
            in accumulated.items()
            if int(row_descriptor) == descriptor_index
            and abs(coefficient) > float(imag_tol)
        ]
        descriptor_rows.sort(key=lambda item: item[0])
        rows.extend(
            (descriptor_index, counts, coefficient)
            for counts, coefficient in descriptor_rows
        )
        offsets.append(len(rows))
    monomial_counts = torch.tensor(
        [row[1] for row in rows],
        dtype=torch.long,
        device=table_device,
    )
    if not rows:
        monomial_counts = torch.empty(
            (0, channel_count),
            dtype=torch.long,
            device=table_device,
        )
    output_index = torch.tensor(
        [row[0] for row in rows],
        dtype=torch.long,
        device=table_device,
    )
    raw_values = [
        row[2]
        if output_dtype in (torch.complex64, torch.complex128)
        else row[2].real
        for row in rows
    ]
    monomial_value = torch.tensor(
        raw_values,
        dtype=output_dtype,
        device=table_device,
    )
    grouped = (
        monomial_counts,
        torch.tensor(offsets, dtype=torch.long, device=table_device),
        output_index,
        monomial_value,
    )
    shared = _shared_symmetric_power_table(
        [row[1] for row in rows],
        offsets,
        [row[0] for row in rows],
        monomial_value,
        channel_count,
        table_device,
    )
    table = (grouped, shared, output_dtype)
    _PRODUCT_PLAN_COUNT_TABLE_CACHE[key] = table
    return table


def symmetric_power_product_plan_contraction(
    input,
    plan,
    *,
    backend="auto",
    imag_tol=1.0e-12,
):
    """Evaluate a compiler-owned symmetric-power descriptor plan."""

    from ye3t.runtime.execution_plan import (
        symmetric_power_monomial_contraction,
        symmetric_power_shared_monomial_contraction,
    )

    input = torch.as_tensor(input)
    if input.ndim < 1:
        raise ValueError(
            "symmetric-power product-plan input requires a channel axis"
        )
    if int(input.shape[-1]) != int(plan.channel_count):
        raise ValueError(
            "symmetric-power plan channel_count does not match input"
        )
    grouped, shared, output_dtype = _symmetric_power_product_plan_count_table(
        plan,
        device=input.device,
        dtype=input.dtype,
        imag_tol=imag_tol,
    )
    converted_input = input.to(dtype=output_dtype)
    if _use_shared_symmetric_power_table(
        converted_input,
        grouped,
        shared,
    ):
        return symmetric_power_shared_monomial_contraction(
            converted_input,
            *shared,
            backend=backend,
        )
    monomial_counts, output_offsets, output_index, monomial_value = grouped
    return symmetric_power_monomial_contraction(
        converted_input,
        monomial_counts,
        output_offsets,
        output_index,
        monomial_value,
        backend=backend,
    )


def symmetric_power_product_plan_batched_adjoint(
    output_adjoint,
    input,
    plan,
    *,
    backend="auto",
    imag_tol=1.0e-12,
):
    """Evaluate many compiler-plan VJPs over one physical input batch."""

    from ye3t.runtime.execution_plan import (
        symmetric_power_shared_monomial_batched_adjoint,
    )

    input = torch.as_tensor(input)
    if input.ndim != 2:
        raise ValueError(
            "symmetric-power batched adjoint input must be two-dimensional"
        )
    if int(input.shape[1]) != int(plan.channel_count):
        raise ValueError(
            "symmetric-power plan channel_count does not match input"
        )
    grouped, shared, output_dtype = _symmetric_power_product_plan_count_table(
        plan,
        device=input.device,
        dtype=input.dtype,
        imag_tol=imag_tol,
    )
    del grouped
    converted_input = input.to(dtype=output_dtype)
    converted_adjoint = torch.as_tensor(
        output_adjoint,
        dtype=output_dtype,
        device=input.device,
    )
    if converted_adjoint.ndim != 3:
        raise ValueError(
            "output_adjoint must have shape [seed, batch, descriptor]"
        )
    if int(converted_adjoint.shape[1]) != int(input.shape[0]):
        raise ValueError("output_adjoint batch must match input")
    if int(converted_adjoint.shape[2]) != int(plan.descriptor_count):
        raise ValueError(
            "output_adjoint descriptor width does not match the plan"
        )
    return symmetric_power_shared_monomial_batched_adjoint(
        converted_adjoint,
        converted_input,
        *shared,
        backend=backend,
    )


def symmetric_power_monomials(x, monomial_table_or_indices):
    """Materialize monomial products for a symmetric-power table or index tensor."""
    if isinstance(monomial_table_or_indices, SymmetricPowerMonomialTable):
        monomial_indices = monomial_table_or_indices.monomial_indices
    else:
        monomial_indices = monomial_table_or_indices
    monomial_indices = monomial_indices.to(device=x.device)
    if monomial_indices.numel() == 0:
        return torch.empty((*x.shape[:-1], 0), dtype=x.dtype, device=x.device)
    gathered = x[..., monomial_indices]
    return gathered.prod(dim=-1)


def _symmetric_power_count_monomials(x, count_table):
    """Materialize monomial products from grouped per-component exponents."""
    counts = count_table.monomial_counts.to(device=x.device)
    if counts.numel() == 0:
        return torch.empty((*x.shape[:-1], 0), dtype=x.dtype, device=x.device)
    flat = x.reshape(-1, x.shape[-1])
    monomials = torch.ones((flat.shape[0], counts.shape[0]), dtype=x.dtype, device=x.device)
    for component in range(counts.shape[1]):
        exponents = counts[:, component]
        active = exponents > 0
        if bool(active.any()):
            monomials[:, active] = monomials[:, active] * flat[:, component:component + 1].pow(exponents[active])
    return monomials.reshape(*x.shape[:-1], counts.shape[0])


def symmetric_square_reference_real_tesseral(x, L, output_L):
    """Evaluate a symmetric square with the ordinary CG reference path."""
    from ye3t.runtime.native import NativeYE3TOperatorModule

    L, output_L = _validate_symmetric_square_inputs(x, L, output_L)
    return NativeYE3TOperatorModule._couple_single(x, x, L, L, output_L)


def _reshape_symmetric_power_output(out, original_shape, multiplicity, output_L):
    out_dim = 2 * int(output_L) + 1
    if int(multiplicity) == 1:
        if int(output_L) == 0:
            if original_shape:
                return out.reshape(*original_shape)
            return out.reshape(())
        return out.reshape(*original_shape, out_dim)
    if int(output_L) == 0:
        return out.reshape(*original_shape, int(multiplicity))
    return out.reshape(*original_shape, int(multiplicity), out_dim)


def symmetric_power_reference_real_tesseral(x, power, L, output_L):
    """Evaluate ``Sym^power(V_L) -> V_output_L`` with the direct product evaluator."""

    return symmetric_power_product_evaluator_real_tesseral(x, power, L, output_L)


def symmetric_power_kernel_real_tesseral(x, power, L, output_L):
    """Evaluate a symmetric power with a folded real-basis monomial kernel."""
    from ye3t.runtime.execution_plan import (
        symmetric_power_monomial_contraction,
        symmetric_power_shared_monomial_contraction,
    )

    power, L, output_L = _validate_symmetric_power_inputs(x, power, L, output_L)
    original_shape = tuple(x.shape[:-1])
    flat = x.reshape(-1, x.shape[-1])
    out_dim = 2 * output_L + 1
    use_wide_accumulation = (
        flat.dtype == torch.float32
        and int(power) >= _FP32_WIDE_ACCUMULATION_MIN_POWER
    )
    table_dtype = torch.float64 if use_wide_accumulation else x.dtype
    table = _symmetric_power_count_table(
        power,
        L,
        output_L,
        device=x.device,
        dtype=table_dtype,
    )
    contraction_input = flat
    contraction_values = table.monomial_value
    if use_wide_accumulation:
        contraction_input = flat.to(dtype=torch.float64)
    out = symmetric_power_monomial_contraction(
        contraction_input,
        table.monomial_counts,
        table.output_offsets,
        table.output_index,
        contraction_values,
        backend="auto",
    )
    if use_wide_accumulation:
        out = out.to(dtype=flat.dtype)
    if int(out.shape[1]) != int(table.multiplicity * out_dim):
        raise RuntimeError(
            "native symmetric-power output width does not match its table"
        )
    return _reshape_symmetric_power_output(out, original_shape, table.multiplicity, output_L)


def symmetric_power_product_evaluator_real_tesseral(x, power, L, output_L):
    """Evaluate a symmetric power with a grouped monomial product evaluator."""
    power, L, output_L = _validate_symmetric_power_inputs(x, power, L, output_L)
    table = symmetric_power_monomial_table(power, L, output_L, device=x.device, dtype=x.dtype)
    original_shape = tuple(x.shape[:-1])
    flat = x.reshape(-1, x.shape[-1])
    if table.term_count == 0:
        out = torch.zeros(
            (flat.shape[0], table.multiplicity * (2 * output_L + 1)),
            dtype=x.dtype,
            device=x.device,
        )
        return _reshape_symmetric_power_output(out, original_shape, table.multiplicity, output_L)

    monomials = torch.ones((flat.shape[0], table.term_count), dtype=x.dtype, device=x.device)
    for slot in range(power):
        monomials = monomials * flat[:, table.monomial_indices[:, slot]]
    weighted = monomials * table.monomial_value
    out_dim = 2 * output_L + 1
    output_index = table.output_basis * out_dim + table.output_m
    columns = []
    for idx in range(table.multiplicity * out_dim):
        mask = output_index == int(idx)
        if bool(mask.any()):
            columns.append(weighted[:, mask].sum(dim=1))
        else:
            columns.append(torch.zeros(flat.shape[0], dtype=x.dtype, device=x.device))
    out = torch.stack(columns, dim=1)
    return _reshape_symmetric_power_output(out, original_shape, table.multiplicity, output_L)


def _select_symmetric_power_auto_evaluator(
    x,
    power,
    L,
    output_L,
):
    del L, output_L
    from ye3t.runtime.execution_plan import (
        native_execution_plan_capabilities,
    )

    require_native = os.environ.get(
        "YE3T_REQUIRE_NATIVE",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if require_native:
        return "native_monomial"
    capabilities = native_execution_plan_capabilities()
    if x.device.type == "cuda":
        if (
            "symmetric_power_monomial"
            in capabilities["cuda_operations"]
        ):
            return "native_monomial"
        return "product_evaluator"
    batch_size = int(x.numel()) // int(x.shape[-1])
    max_batch = capabilities["symmetric_power_cpu_auto_max_batch"]
    min_power = capabilities["symmetric_power_cpu_auto_min_power"]
    if (
        x.device.type == "cpu"
        and capabilities["prebuilt_extension"]
        and (
            int(power) >= int(min_power)
            or batch_size <= int(max_batch)
        )
    ):
        return "native_monomial"
    return "product_evaluator"


def symmetric_power_real_tesseral(x, power, L, output_L, *, optimization_policy = "auto"):
    """Evaluate ``Sym^power(V_L) -> V_output_L`` with optional monomial kernels."""
    optimization_policy = _validate_optimization_policy(optimization_policy)
    if optimization_policy == "off":
        return symmetric_power_reference_real_tesseral(x, power, L, output_L)
    if optimization_policy == "product_evaluator":
        return symmetric_power_product_evaluator_real_tesseral(x, power, L, output_L)
    if (
        optimization_policy == "auto"
        and _select_symmetric_power_auto_evaluator(
            x,
            power,
            L,
            output_L,
        )
        == "product_evaluator"
    ):
        return symmetric_power_product_evaluator_real_tesseral(x, power, L, output_L)
    return symmetric_power_kernel_real_tesseral(x, power, L, output_L)


def symmetric_power_outputs_real_tesseral(
    x,
    power,
    L,
    output_Ls=None,
    *,
    optimization_policy="auto",
):
    """Evaluate selected ``Sym^power(V_L)`` outputs in one grouped schedule."""

    power = _validate_symmetric_power(power)
    L = int(L)
    if x.shape[-1] != 2 * L + 1:
        raise ValueError(
            f"Expected final dimension {2 * L + 1} for L={L}, "
            f"got {x.shape[-1]}."
        )
    if output_Ls is None:
        output_Ls = allowed_symmetric_power_outputs(power, L)
    output_Ls = tuple(int(value) for value in output_Ls)
    optimization_policy = _validate_optimization_policy(
        optimization_policy
    )
    if not output_Ls:
        return {}
    use_product_evaluator = optimization_policy in {
        "off",
        "product_evaluator",
    }
    if (
        optimization_policy == "auto"
        and _select_symmetric_power_auto_evaluator(
            x,
            power,
            L,
            output_Ls[0],
        )
        == "product_evaluator"
    ):
        use_product_evaluator = True
    if use_product_evaluator:
        policy = (
            "off"
            if optimization_policy == "off"
            else "product_evaluator"
        )
        return {
            output_L: symmetric_power_real_tesseral(
                x,
                power,
                L,
                output_L,
                optimization_policy=policy,
            )
            for output_L in output_Ls
        }

    from ye3t.runtime.execution_plan import (
        symmetric_power_monomial_contraction,
        symmetric_power_shared_monomial_contraction,
    )

    grouped, shared, output_slices = _combined_symmetric_power_count_table(
        power,
        L,
        output_Ls,
        device=x.device,
        dtype=x.dtype,
    )
    original_shape = tuple(x.shape[:-1])
    flat = x.reshape(-1, x.shape[-1])
    if _use_shared_symmetric_power_table(flat, grouped, shared):
        packed = symmetric_power_shared_monomial_contraction(
            flat,
            *shared,
            backend="auto",
        )
    else:
        packed = symmetric_power_monomial_contraction(
            flat,
            *grouped,
            backend="auto",
        )
    outputs = {}
    for output_L, multiplicity, start, stop in output_slices:
        outputs[int(output_L)] = _reshape_symmetric_power_output(
            packed[:, int(start):int(stop)],
            original_shape,
            multiplicity,
            output_L,
        )
    return outputs


def symmetric_square_kernel_real_tesseral(x, L, output_L):
    """Evaluate a symmetric square with the folded pair-product kernel."""
    L, output_L = _validate_symmetric_square_inputs(x, L, output_L)
    table = symmetric_square_pair_table(L, output_L, device=x.device, dtype=x.dtype)
    original_shape = tuple(x.shape[:-1])
    flat = x.reshape(-1, x.shape[-1])
    pair_terms = flat.index_select(1, table.pair_index)
    pair_count = int(table.pair_count)
    pair_products = pair_terms[:, :pair_count] * pair_terms[:, pair_count:]
    if int(flat.shape[0]) >= 64:
        out = pair_products @ _symmetric_square_dense_table(L, output_L, x)
    else:
        products = pair_products * table.pair_value
        out = torch.zeros(
            (flat.shape[0], 2 * output_L + 1),
            dtype=x.dtype,
            device=x.device,
        )
        out.index_add_(1, table.pair_out, products)
    if output_L == 0:
        if original_shape:
            return out.reshape(*original_shape)
        return out.reshape(())
    return out.reshape(*original_shape, 2 * output_L + 1)


def symmetric_square_real_tesseral(x, L, output_L, *, optimization_policy = "auto"):
    """Evaluate ``Sym^2(V_L) -> V_output_L`` with an optional folded kernel."""
    optimization_policy = _validate_optimization_policy(optimization_policy)
    if optimization_policy == "off":
        return symmetric_square_reference_real_tesseral(x, L, output_L)
    return symmetric_square_kernel_real_tesseral(x, L, output_L)


def symmetric_square_all_real_tesseral(x, L, *, optimization_policy = "auto"):
    """Evaluate every irrep in ``Sym^2(V_L)`` for one repeated channel tensor."""
    return {
        output_L: symmetric_square_real_tesseral(x, int(L), output_L, optimization_policy=optimization_policy)
        for output_L in allowed_symmetric_square_outputs(int(L))
    }


def symmetric_square_channels_real_tesseral(x, L, output_L, *, optimization_policy = "auto"):
    """Evaluate ``Sym^2(V_L)`` channelwise for ``x[..., channels, m]``."""
    L = int(L)
    if x.shape[-1] != 2 * L + 1:
        raise ValueError(f"Expected final dimension {2 * L + 1} for L={L}, got {x.shape[-1]}.")
    return symmetric_square_real_tesseral(x, L, int(output_L), optimization_policy=optimization_policy)


def symmetric_cube_real_tesseral(x, L, output_L, *, optimization_policy = "auto"):
    """Evaluate ``Sym^3(V_L) -> V_output_L`` with optional monomial kernels."""
    return symmetric_power_real_tesseral(
        x,
        3,
        L,
        output_L,
        optimization_policy=optimization_policy,
    )


def symmetric_fourth_power_real_tesseral(x, L, output_L, *, optimization_policy = "auto"):
    """Evaluate ``Sym^4(V_L) -> V_output_L`` with optional monomial kernels."""
    return symmetric_power_real_tesseral(
        x,
        4,
        L,
        output_L,
        optimization_policy=optimization_policy,
    )


__all__ = [
    "SymmetricSquarePairTable",
    "SymmetricPowerMonomialTable",
    "allowed_symmetric_square_outputs",
    "allowed_symmetric_power_outputs",
    "symmetric_power_output_multiplicity",
    "symmetric_square_pair_table",
    "symmetric_power_monomial_table",
    "symmetric_power_monomials",
    "symmetric_square_reference_real_tesseral",
    "symmetric_square_kernel_real_tesseral",
    "symmetric_power_reference_real_tesseral",
    "symmetric_power_kernel_real_tesseral",
    "symmetric_power_product_evaluator_real_tesseral",
    "symmetric_power_product_plan_batched_adjoint",
    "symmetric_power_product_plan_contraction",
    "symmetric_power_real_tesseral",
    "symmetric_power_outputs_real_tesseral",
    "symmetric_square_real_tesseral",
    "symmetric_square_all_real_tesseral",
    "symmetric_square_channels_real_tesseral",
    "symmetric_cube_real_tesseral",
    "symmetric_fourth_power_real_tesseral",
]
