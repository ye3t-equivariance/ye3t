"""Differentiable products with separate Young, angular and channel maps."""

import torch
import sympy as sp


class _CarrierProduct(torch.nn.Module):
    def __init__(self, compiled, *, dtype=None, device=None, backend="auto"):
        super().__init__()
        self.compiled = compiled
        self.calls = 0
        if backend not in {"auto", "torch", "native"}:
            raise ValueError("Carrier product backend must be auto, torch or native")
        self.requested_backend = backend
        self.executed_backend = "not_executed"
        self._sparse = None
        self.sparse_terms = 0
        self.native_reason = "not_evaluated"
        dtype = torch.get_default_dtype() if dtype is None else dtype
        from ye3t.runtime._carrier_product_sparse import sparse_schedule_size
        self.sparse_terms = sparse_schedule_size(compiled, dtype.is_complex)
        left, right = (compiled["request"][name] for name in ("left", "right"))
        output = compiled["output_layout"]
        young = torch.zeros((compiled["placement_count"], compiled["young_mode_count"],
            left["tableau_count"], right["tableau_count"], output["tableau_count"]), dtype=dtype, device=device)
        angular = torch.zeros((left["magnetic_count"], right["magnetic_count"], output["magnetic_count"]), dtype=dtype, device=device)
        for p, h, i, j, t, value in compiled["young"]:
            young[p, h, i, j, t] = float(sp.sympify(value))
        for m, n, M, value in compiled["angular"]:
            angular[m, n, M] = float(sp.sympify(value))
        self.register_buffer("young", young)
        self.register_buffer("angular", angular)
        indices, swapped, scales = [], [], []
        nh, nr = compiled["young_mode_count"], right["channel_count"]
        for row in compiled["channels"]:
            a, b, h = (row[name] for name in ("left_channel", "right_channel", "young_copy"))
            indices.append((a*nr+b)*nh+h)
            swapped.append((b*nr+a)*nh+h if row["exchange"] else (a*nr+b)*nh+h)
            scales.append((row["exchange"] if a != b else 0)/2**.5)
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long, device=device))
        self.register_buffer("swapped", torch.tensor(swapped, dtype=torch.long, device=device))
        self.register_buffer("swap_scale", torch.tensor(scales, dtype=dtype, device=device))
        self.register_buffer("direct_scale", torch.tensor([2**-.5 if row["exchange"] and row["left_channel"] != row["right_channel"] else 1
            for row in compiled["channels"]], dtype=dtype, device=device))
        self.role_scale = float(sp.sympify(compiled["role_scale"]))

    def forward(self, left, right, *, placements=None):
        request = self.compiled["request"]
        if request["argument_ids"][0] == request["argument_ids"][1] and left is not right:
            raise ValueError("A repeated argument must bind the same runtime tensor")
        for value, name in ((left, "left"), (right, "right")):
            layout = request[name]
            if tuple(value.shape[-3:]) != (layout["channel_count"], layout["tableau_count"], layout["magnetic_count"]):
                raise ValueError("Product input differs from its complete carrier layout")
            if value.dtype != self.angular.dtype or value.device != self.angular.device:
                raise ValueError("Product inputs must match the bound dtype and device")
        if left.shape[:-3] != right.shape[:-3]:
            raise ValueError("Product arguments must share batch and placement axes")
        if request["operation"] == "disjoint_induction":
            if (placements != self.compiled["placements"] or left.ndim < 4
                    or left.shape[-4] != self.compiled["placement_count"]):
                raise ValueError("Induction requires every compiler-ordered placement exactly once")
        else:
            if placements is not None:
                raise ValueError("Same-action products have no induced placement axis")
        if self._use_native(left):
            result = self._sparse(left, right)
            self.calls += 1
            self.executed_backend = "triton_deterministic_sparse_bilinear"
            return result
        if request["operation"] == "disjoint_induction":
            coupled = torch.einsum("...pcim,...pdjn,mno,phijk->...cdhko", left, right, self.angular, self.young)
        else:
            coupled = torch.einsum("...cim,...djn,mno,hijk->...cdhko", left, right, self.angular, self.young[0])
        flat = coupled.flatten(-5, -3)
        result = flat.index_select(-3, self.indices)*self.direct_scale[:, None, None]
        if request["argument_ids"][0] == request["argument_ids"][1]:
            result = result + flat.index_select(-3, self.swapped)*self.swap_scale[:, None, None]
        self.calls += 1
        self.executed_backend = "torch_factorized_young_angular_product"
        return result*self.role_scale

    def _use_native(self, left):
        from ye3t.runtime._carrier_product_sparse import _SparseCarrierProduct
        if self.requested_backend == "torch":
            self.native_reason = "torch_requested"
            return False
        if left.device.type != "cuda":
            self.native_reason = "device_is_not_cuda"
        elif left.dtype not in {torch.float32, torch.float64, torch.complex64, torch.complex128}:
            self.native_reason = "dtype_requires_factorized_torch"
        elif self.sparse_terms > 262144 and self.requested_backend == "auto":
            # This bounds only the optional expanded sparse schedule. Every
            # coordinate remains available through the factorized evaluator.
            self.native_reason = "sparse_expansion_exceeds_auto_term_limit"
        else:
            from ye3t.backends._deterministic_bilinear import triton
            if triton is not None:
                if self._sparse is None:
                    self._sparse = _SparseCarrierProduct(self.compiled, dtype=left.dtype, device=left.device)
                self.native_reason = "eligible"
                return True
            self.native_reason = "triton_unavailable"
        if self.requested_backend == "native":
            raise RuntimeError("Native carrier product unavailable: " + self.native_reason)
        return False

    def _apply(self, fn):
        self._sparse = None
        self.executed_backend = "not_executed"
        self.native_reason = "not_evaluated"
        result = super()._apply(fn)
        from ye3t.runtime._carrier_product_sparse import sparse_schedule_size
        self.sparse_terms = sparse_schedule_size(self.compiled, self.angular.is_complex())
        return result

    def runtime_report(self):
        return {"backend": self.executed_backend, "requested_backend": self.requested_backend,
                "auto_sparse_term_limit": 262144,
                "native_selection": self.native_reason, "sparse_terms": self.sparse_terms,
                "sparse_table_bytes": self._sparse.table_bytes if self._sparse is not None else 0,
                "sparse_launches_forward_left_right": self._sparse.last_launches if self._sparse is not None else None,
                "executed_calls": self.calls,
                "plan_hash": self.compiled["self_hash"], "coefficient_backend": self.compiled["coefficient_backend"],
                "full_joint_matrix": False, "derivatives": ["value", "vjp", "double_backward"]}
