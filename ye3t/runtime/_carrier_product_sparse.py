"""CUDA schedules expanded directly from canonical nonzero carrier maps.

Complex coordinates use interleaved real/imaginary storage. This changes no
angular convention or phase. The existing deterministic bilinear kernels
provide values and derivatives through order two, including force training.
"""

from math import prod

import sympy as sp
import torch


def sparse_schedule_size(compiled, complex_storage):
    counts = [0] * compiled["young_mode_count"]
    for _, h, *_ in compiled["young"]:
        counts[h] += 1
    terms = sum(counts[row["young_copy"]] *
        (2 if row["exchange"] and row["left_channel"] != row["right_channel"] else 1)
        for row in compiled["channels"]) * len(compiled["angular"])
    return terms * (4 if complex_storage else 1)


class _SparseCarrierProduct(torch.nn.Module):
    def __init__(self, compiled, *, dtype, device):
        super().__init__()
        from ye3t.backends._deterministic_bilinear import bilinear_plan
        self.complex_storage = dtype.is_complex
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        layouts = [compiled["request"][name] for name in ("left", "right")]
        output = compiled["output_layout"]
        self.output_shape = tuple(output[key] for key in ("channel_count", "tableau_count", "magnetic_count"))
        self.induced = compiled["request"]["operation"] == "disjoint_induction"
        scale = 2 if self.complex_storage else 1
        widths = [scale * compiled["placement_count"] * prod(layout[key]
            for key in ("channel_count", "tableau_count", "magnetic_count")) for layout in layouts]
        widths.append(scale * prod(self.output_shape))
        young = [[] for _ in range(compiled["young_mode_count"])]
        for p, h, i, j, t, value in compiled["young"]:
            young[h].append((p, i, j, t, sp.sympify(value)))
        angular = [(m, n, M, sp.sympify(value)) for m, n, M, value in compiled["angular"]]
        role_scale = sp.sympify(compiled["role_scale"])
        indices = [[], [], []]
        coefficients = []
        for channel, row in enumerate(compiled["channels"]):
            a, b = row["left_channel"], row["right_channel"]
            exchange = bool(row["exchange"] and a != b)
            pairs = [(a, b, 1 / sp.sqrt(2) if exchange else sp.Integer(1))]
            if exchange:
                pairs.append((b, a, row["exchange"] / sp.sqrt(2)))
            for p, i, j, t, y in young[row["young_copy"]]:
                for m, n, M, cg in angular:
                    for ca, cb, weight in pairs:
                        value = y * cg * weight * role_scale
                        if value.is_real is not True:
                            raise ValueError("Sparse carrier lowering requires real canonical fixed maps")
                        li = ((p * layouts[0]["channel_count"] + ca) * layouts[0]["tableau_count"] + i) * layouts[0]["magnetic_count"] + m
                        ri = ((p * layouts[1]["channel_count"] + cb) * layouts[1]["tableau_count"] + j) * layouts[1]["magnetic_count"] + n
                        oi = (channel * output["tableau_count"] + t) * output["magnetic_count"] + M
                        terms = ((2*li, 2*ri, 2*oi, 1), (2*li+1, 2*ri+1, 2*oi, -1),
                                 (2*li, 2*ri+1, 2*oi+1, 1), (2*li+1, 2*ri, 2*oi+1, 1)) if self.complex_storage else ((li, ri, oi, 1),)
                        for l, r, o, sign in terms:
                            for target, index in zip(indices, (l, r, o)):
                                target.append(index)
                            coefficients.append(sign * float(value))
        if len(coefficients) != sparse_schedule_size(compiled, self.complex_storage):
            raise ArithmeticError("Sparse carrier schedule differs from its canonical nonzero term count")
        tables, self.sizes = bilinear_plan(*indices, *widths)
        # Short CSR sums leave most of a 32-term tile inactive. Wider row
        # tiles amortize the same canonical map over more occurrences. Select
        # each adjoint separately because its CSR degree can be much larger.
        # The reduction order remains deterministic for each selected launch.
        self.launches = tuple((128, 4) if int((tables[3+5*axis][1:]-tables[3+5*axis][:-1]).max()) <= 24
                              else (8, 32) for axis in (2, 0, 1))
        self.last_launches = None
        for index, table in enumerate(tables):
            self.register_buffer("table_" + str(index), table.to(device), persistent=False)
        self.table_count = len(tables)
        # One coefficient vector per destination axis, in that axis's event
        # order, so every launch streams its events contiguously.
        coefficient = torch.tensor(coefficients, dtype=real_dtype)
        for axis in range(3):
            self.register_buffer("coefficient_" + str(axis), coefficient[tables[4+5*axis]].to(device), persistent=False)
        self.table_bytes = sum(buffer.numel()*buffer.element_size() for buffer in self.buffers())

    def forward(self, left, right):
        from ye3t.backends._deterministic_bilinear import deterministic_bilinear
        batch_shape = left.shape[:-(4 if self.induced else 3)]
        batch = prod(batch_shape)
        def pack(value, width):
            value = value.resolve_conj().contiguous()
            return (torch.view_as_real(value) if self.complex_storage else value).reshape(batch, width)
        result = deterministic_bilinear(pack(left, self.sizes[0]), pack(right, self.sizes[1]),
            tuple(getattr(self, "coefficient_"+str(axis)) for axis in range(3)),
            tuple(getattr(self, "table_"+str(i)) for i in range(self.table_count)), self.sizes,
            launches=self.launches)
        self.last_launches = self.launches if batch >= 128 else ((8, 32),)*3
        if self.complex_storage:
            result = torch.view_as_complex(result.reshape(batch, prod(self.output_shape), 2))
        return result.reshape(*batch_shape, *self.output_shape)
