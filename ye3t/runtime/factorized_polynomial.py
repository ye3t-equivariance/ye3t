"""Shared staged polynomial evaluation and analytical batched pullbacks."""
import torch
from collections import OrderedDict
from math import isfinite

from .execution_plan import (
    _gathered_factorized_angular,
    packed_concatenate,
    symmetric_power_shared_monomial_contraction,
    symmetric_power_shared_monomial_batched_adjoint,
)

# Bound additional batching workspace. An individual existing program may be
# larger; batching never combines it with another program beyond this target.
_ANGULAR_BATCH_WORKSPACE_BYTES = 8 * 1024 * 1024
# Large fixed maps retain compiler coordinates instead of quadratic storage.
_DENSE_OUTPUT_MAP_BYTES = 8 * 1024 * 1024


class _FactorizedPolynomialRuntime(torch.nn.Module):
    """Bind compiler-owned stages once; retain no geometry in persistent state."""

    def __init__(self, schedule, *, dtype=torch.float64, backend="native", path_basis="real_tesseral"):
        super().__init__()
        self.backend = backend
        if path_basis not in {"real_tesseral", "complex_condon_shortley"}:
            raise ValueError("Unknown internal path storage basis")
        self.path_basis = path_basis
        if backend not in {"native", "reference"}:
            raise ValueError("Factorized backend must be native or reference")
        self.schedule = schedule
        self.programs = []
        self.path_programs = {}
        self.path_output_shapes = {}
        self.compact_output_versions = {}
        self.path_bindings = {}
        self.local_path_bindings = {}
        self.path_batch_plans = {}
        self.local_group_plans = {}
        self.fixed_output_bindings = {}
        self.packed_output_plans = OrderedDict()
        self.packed_output_calls = 0
        self.packed_linear_output_calls = 0
        self.packed_output_plan_builds = 0
        self.input_bindings = [tuple(program["input_indices"]) for program in schedule["programs"]]
        self.execution_name = "factorized_polynomial_" + backend
        for p, program in enumerate(schedule["programs"]):
            if any(not 0 <= i < schedule["input_dimension"] for i in program["input_indices"]):
                raise ValueError("Factorized input binding is outside the source dimension")
            self.register_buffer(f"inputs_{p}", torch.tensor(program["input_indices"], dtype=torch.long))
            self.register_buffer(f"outputs_{p}", torch.tensor(program["selected_outputs"], dtype=torch.long))
            if program["selected_outputs"] == list(range(program["schedule"]["output_dimension"])):
                self._bind_fixed_output(f"outputs_{p}", None)
            if program["schedule"]["schema"] in {
                    "ye3t_factorized_shell_schedule_v2", "ye3t_factorized_cauchy_role_schedule_v1"}:
                path = program["schedule"]["cauchy_paths"]
                self.path_programs[p] = path
                width = len(path["routes"])*(2*path["output_L"]+1)
                target_width = program["schedule"]["output_dimension"]
                if (sum(path["role_dimension"]*(2*ell+1) for ell in path["input_Ls"])
                        != len(program["input_indices"])
                        or any(not 0 <= i < target_width for i in program["selected_outputs"])):
                    raise ValueError("Shared path input/output dimensions do not compose")
                entries = program["schedule"]["output_map"]
                for source, target, coefficient in entries:
                    if not (0 <= source < width and 0 <= target < target_width and isfinite(coefficient)):
                        raise ValueError("Invalid shared path output map")
                self.path_output_shapes[p] = (width, target_width)
                if width*target_width*torch.empty((), dtype=dtype).element_size() > _DENSE_OUTPUT_MAP_BYTES:
                    rows, columns, coefficients = self._compiler_output_coordinates(p, dtype)
                    for name, value in zip(self._compact_output_names(p), (rows, columns, coefficients), strict=True):
                        self.register_buffer(name, value)
                else:
                    output_map = torch.zeros((width, target_width), dtype=dtype)
                    for source, target, coefficient in entries:
                        output_map[source, target] += coefficient
                    self.register_buffer(f"path_output_map_{p}", output_map)
                    self._bind_path_output(p, output_map)
                self.programs.append(None)
                self.execution_name = "shared_young_cg_native" if backend == "native" else "shared_young_cg_aten"
                continue
            stages = program["schedule"]["stages"]
            previous_width = len(program["input_indices"])
            for s, stage in enumerate(stages):
                if stage["input_dimension"] != previous_width:
                    raise ValueError("Factorized polynomial stage dimensions do not compose")
                previous_width = stage["output_dimension"]
                term_offsets = stage["term_offsets"]
                output_offsets = stage["output_offsets"]
                if (not term_offsets or term_offsets[0] != 0 or
                        sorted(term_offsets) != list(term_offsets) or
                        term_offsets[-1] != len(stage["term_components"]) or
                        len(stage["term_components"]) != len(stage["term_exponents"]) or
                        len(output_offsets) != previous_width+1 or output_offsets[0] != 0 or
                        sorted(output_offsets) != list(output_offsets) or
                        output_offsets[-1] != len(stage["coefficient_values"]) or
                        len(stage["coefficient_terms"]) != output_offsets[-1] or
                        len(stage["coefficient_outputs"]) != output_offsets[-1]):
                    raise ValueError("Factorized sparse table offsets are invalid")
                if (any(not 0 <= i < stage["input_dimension"] for i in stage["term_components"]) or
                        any(e < 1 or int(e) != e for e in stage["term_exponents"]) or
                        any(not 0 <= i < len(term_offsets)-1 for i in stage["coefficient_terms"]) or
                        any(not isfinite(c) for c in stage["coefficient_values"]) or
                        any(stage["coefficient_outputs"][j] != i for i in range(previous_width)
                            for j in range(output_offsets[i], output_offsets[i+1]))):
                    raise ValueError("Factorized sparse table entries are invalid")
                counts = torch.zeros((len(stage["term_offsets"])-1, stage["input_dimension"]), dtype=torch.long)
                for row, (begin, end) in enumerate(zip(stage["term_offsets"][:-1], stage["term_offsets"][1:])):
                    for entry in range(begin, end):
                        counts[row, stage["term_components"][entry]] += stage["term_exponents"][entry]
                self.register_buffer(f"counts_{p}_{s}", counts)
                for name in ("output_offsets", "coefficient_terms", "coefficient_outputs", "coefficient_values"):
                    self.register_buffer(f"{name}_{p}_{s}", torch.tensor(stage[name],
                        dtype=dtype if name == "coefficient_values" else torch.long))
            if any(not 0 <= i < previous_width for i in program["selected_outputs"]):
                raise ValueError("Factorized output binding is outside the source dimension")
            self.programs.append(stages)
        if sum(len(program["selected_outputs"]) for program in schedule["programs"]) != schedule["output_dimension"]:
            raise ValueError("Factorized polynomial output selection has the wrong width")

    def __getattr__(self, name):
        # Explicit inspection retains the historical dense-buffer interface.
        # Execution accesses compact buffers directly and never triggers this.
        if name.startswith("path_output_map_") and name.removeprefix("path_output_map_").isdigit():
            p = int(name.removeprefix("path_output_map_"))
            buffers = self.__dict__.get("_buffers", {})
            if name not in buffers and f"path_output_values_{p}" in buffers:
                rows, columns, coefficients = self._compact_output_tables(p)
                matrix = coefficients.new_zeros(self.path_output_shapes[p])
                matrix.index_put_((rows, columns), coefficients, accumulate=True)
                for key in self._compact_output_names(p):
                    del buffers[key]
                self.compact_output_versions.pop(p, None)
                self.packed_output_plans.clear()
                self.register_buffer(name, matrix)
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name.startswith("path_output_map_") and name.removeprefix("path_output_map_").isdigit():
            p = int(name.removeprefix("path_output_map_"))
            buffers = self.__dict__.get("_buffers", {})
            if name not in buffers and f"path_output_values_{p}" in buffers:
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != self.path_output_shapes[p]:
                    raise ValueError("Replacement shared path output map has the wrong shape")
                for key in self._compact_output_names(p):
                    del buffers[key]
                self.compact_output_versions.pop(p, None)
                self.packed_output_plans.clear()
                self.register_buffer(name, value)
                return
        super().__setattr__(name, value)

    def _compact_output_names(self, p):
        return tuple(f"path_output_{name}_{p}" for name in ("rows", "columns", "values"))

    def _output_buffers(self, p):
        names = ((f"path_output_map_{p}",) if f"path_output_map_{p}" in self._buffers
                 else self._compact_output_names(p))
        return tuple(self._buffers[name] for name in names)

    def _compiler_output_coordinates(self, p, dtype):
        entries = self.schedule["programs"][p]["schedule"]["output_map"]
        grouped = {}
        for row, column, value in entries:
            grouped.setdefault((row, column), []).append(value)
        coordinates = sorted(grouped)
        values = torch.tensor([grouped[key][0] for key in coordinates], dtype=dtype)
        # Match the old dense map's sequential, dtype-specific accumulation.
        # Keep cancelled coordinates so float/double rebuilds retain shape.
        for index, key in enumerate(coordinates):
            for value in grouped[key][1:]:
                values[index] += value
        return (torch.tensor([key[0] for key in coordinates], dtype=torch.long),
                torch.tensor([key[1] for key in coordinates], dtype=torch.long), values)

    def _compact_output_tables(self, p):
        tables = self._output_buffers(p)
        versions = tuple((id(value), None if torch.is_inference(value) else value._version) for value in tables)
        saved = self.compact_output_versions.get(p)
        if saved is None or saved[0] != versions or any(version is None for _, version in versions):
            rows, columns, values = tables
            height, width = self.path_output_shapes[p]
            if (any(value.ndim != 1 for value in tables) or len(rows) != len(values)
                    or len(columns) != len(values) or rows.dtype != torch.long or columns.dtype != torch.long
                    or rows.device != values.device or columns.device != values.device
                    or not bool(torch.isfinite(values).all())
                    or bool(((rows < 0) | (rows >= height)).any())
                    or bool(((columns < 0) | (columns >= width)).any())):
                raise ValueError("Invalid compact shared path output map")
            # Retain owners as well as versions so replacement cannot reuse an
            # old Python identity and accidentally skip index validation.
            self.compact_output_versions[p] = (versions, tables)
        return tables

    def _apply_output_map(self, p, values, transpose=False):
        buffers = self._output_buffers(p)
        if len(buffers) == 1:
            return values @ (buffers[0].T if transpose else buffers[0])
        rows, columns, coefficients = self._compact_output_tables(p)
        width = self.path_output_shapes[p][0 if transpose else 1]
        if transpose:
            rows, columns = columns, rows
        if self.backend == "native" and not coefficients.requires_grad and not torch.is_inference(coefficients):
            from .execution_plan import _load_extension
            _load_extension()
            source_width = values.shape[-1]
            if not source_width or not width:
                return values.sum(dim=-1, keepdim=True).expand(*values.shape[:-1], width)*0
            identity = torch.arange(source_width, device=values.device)
            return torch.ops.ye3t_runtime.source_analysis(
                values.reshape(-1, source_width).contiguous(), identity, identity, values.new_ones(source_width),
                rows, columns, coefficients, source_width, width).reshape(*values.shape[:-1], width)
        return values.new_zeros((*values.shape[:-1], width)).index_add(
            -1, columns, values.index_select(-1, rows)*coefficients)

    def _apply(self, fn):
        self.packed_output_plans.clear()
        self.compact_output_versions.clear()
        previous_dtypes = {name: value.dtype for name, value in self.named_buffers()}
        bindings = {name: binding[2] for name in tuple(self.fixed_output_bindings)
                    if (binding := self._fixed_output_binding(name)) is not None}
        result = super()._apply(fn)
        self.fixed_output_bindings.clear()
        for name, indices in bindings.items():
            self._bind_fixed_output(name, None if indices is None else indices.to(getattr(self, name).device))
        self.path_bindings.clear()
        self.local_path_bindings.clear()
        self.path_batch_plans.clear()
        self.local_group_plans.clear()
        # These are immutable compiler coefficients, not trainable state. A
        # float32->float64 move must rebuild their precision from the schedule.
        for p, program in enumerate(self.schedule["programs"]):
            if p in self.path_programs:
                if f"path_output_map_{p}" not in self._buffers:
                    name = f"path_output_values_{p}"
                    destination = self._buffers[name]
                    if destination.dtype != previous_dtypes[name]:
                        for key, value in zip(self._compact_output_names(p),
                                              self._compiler_output_coordinates(p, destination.dtype), strict=True):
                            self._buffers[key] = value.to(destination.device)
                    continue
                destination = getattr(self, f"path_output_map_{p}")
                if destination.dtype == previous_dtypes[f"path_output_map_{p}"]:
                    continue
                values = torch.zeros(destination.shape, dtype=destination.dtype)
                for source, target, coefficient in program["schedule"]["output_map"]:
                    values[source, target] += coefficient
                destination.copy_(values.to(destination.device))
                self._bind_path_output(p, values)
            else:
                for s, stage in enumerate(program["schedule"]["stages"]):
                    destination = getattr(self, f"coefficient_values_{p}_{s}")
                    if destination.dtype == previous_dtypes[f"coefficient_values_{p}_{s}"]:
                        continue
                    destination.copy_(torch.tensor(stage["coefficient_values"], dtype=destination.dtype,
                                                   device=destination.device))
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        self.packed_output_plans.clear()
        state, prefix = args[:2]
        # Accept earlier dense states and compact states, including after an
        # explicit dense inspection. Storage is internal, independent of basis.
        for p in self.path_programs:
            name = f"path_output_map_{p}"
            compact = self._compact_output_names(p)
            if prefix+name in state:
                if name not in self._buffers:
                    device = self._buffers[compact[-1]].device
                    dtype = self._buffers[compact[-1]].dtype
                    for key in compact:
                        self._buffers.pop(key, None)
                    self.register_buffer(name, torch.empty(self.path_output_shapes[p], device=device, dtype=dtype))
            elif all(prefix+key in state for key in compact):
                incoming = tuple(state[prefix+key] for key in compact)
                if (any(value.ndim != 1 for value in incoming)
                        or len({value.numel() for value in incoming}) != 1
                        or incoming[0].dtype != torch.long or incoming[1].dtype != torch.long
                        or not incoming[2].is_floating_point()):
                    raise ValueError("Invalid compact shared path output map state")
                destination = self._output_buffers(p)[-1]
                self._buffers.pop(name, None)
                for key in compact:
                    self._buffers[key] = torch.empty(state[prefix+key].shape, device=destination.device,
                        dtype=destination.dtype if key == compact[-1] else torch.long)
        self.compact_output_versions.clear()
        super()._load_from_state_dict(*args, **kwargs)
        self.path_bindings.clear()
        self.local_path_bindings.clear()
        self.path_batch_plans.clear()
        self.local_group_plans.clear()
        # Inspect the loaded buffers once, rather than assuming a checkpoint's
        # maps equal the construction schedule. This also handles assign=True.
        self.fixed_output_bindings.clear()
        for p, program in enumerate(self.schedule["programs"]):
            if p in self.path_programs:
                if f"path_output_map_{p}" in self._buffers:
                    self._bind_path_output(p, getattr(self, f"path_output_map_{p}").detach().cpu())
                else:
                    self._compact_output_tables(p)
            indices = getattr(self, f"outputs_{p}")
            if torch.equal(indices.cpu(), torch.arange(program["schedule"]["output_dimension"])):
                self._bind_fixed_output(f"outputs_{p}", None)

    def _bind_fixed_output(self, name, indices):
        buffer = getattr(self, name)
        if not buffer.requires_grad and not torch.is_inference(buffer):
            self.fixed_output_bindings[name] = (buffer, buffer._version, indices)

    def _fixed_output_binding(self, name):
        """Reuse only the exact fixed buffer inspected at binding time."""
        binding = self.fixed_output_bindings.get(name)
        if binding is not None:
            buffer = getattr(self, name)
            if (buffer is binding[0] and not buffer.requires_grad
                    and not torch.is_inference(buffer) and buffer._version == binding[1]):
                return binding
            # Release old storage after replacement; no coefficient scanning,
            # synchronization or hashing belongs in geometry evaluation.
            self.fixed_output_bindings.pop(name)
        return None

    def _bind_path_output(self, p, values):
        """Inspect the already materialized CPU map, including its dtype rounding."""
        name = f"path_output_map_{p}"
        self.fixed_output_bindings.pop(name, None)
        if not values.shape[1]:
            return
        nonzero = values != 0
        if not bool(torch.all(nonzero.sum(dim=0) == 1)):
            return
        indices = nonzero.to(torch.long).argmax(dim=0)
        if not bool(torch.all(values[indices, torch.arange(values.shape[1])] == 1)):
            return
        identity = values.shape[0] == values.shape[1] and torch.equal(indices, torch.arange(values.shape[1]))
        self._bind_fixed_output(name, None if identity else indices.to(getattr(self, name).device))

    def _select_outputs(self, p, values):
        if self._fixed_output_binding(f"outputs_{p}") is not None:
            return values
        return values.index_select(-1, getattr(self, f"outputs_{p}"))

    def _path_inputs(self, p, values):
        from ye3t.core.tesseral import _real_to_complex_matrix
        from .angular_paths import BoundCauchyPaths

        path = self.path_programs[p]
        real = self.path_basis == "real_tesseral"
        storage_dtype = values.dtype if real else torch.complex128 if values.dtype == torch.float64 else torch.complex64
        key = (p, storage_dtype, values.device, self.path_basis)
        if key not in self.path_bindings:
            indices = self.schedule["programs"][p]["input_indices"]
            bindings, offset = [], 0
            for ell in path["input_Ls"]:
                width = path["role_dimension"]*(2*ell+1)
                bindings.append(tuple(indices[offset:offset+width]))
                offset += width
            self.path_bindings[key] = BoundCauchyPaths(path, dtype=storage_dtype, device=values.device,
                backend=self.backend, basis=self.path_basis, _local_pool=self.local_path_bindings,
                _channel_bindings=bindings)
        channels, forms, offset = {}, {}, 0
        for index, ell in enumerate(path["input_Ls"]):
            width = path["role_dimension"]*(2*ell+1)
            channels[index] = values[..., offset:offset+width].reshape(
                *values.shape[:-1], path["role_dimension"], 2*ell+1)
            if not real:
                forms[index] = _real_to_complex_matrix(ell, values.dtype, values.device)
                channels[index] = channels[index].to(storage_dtype) @ forms[index]
            offset += width
        if offset != values.shape[-1]:
            raise ValueError("Role/angular input bindings have inconsistent dimensions")
        return self.path_bindings[key], channels, forms

    def _path_forward(self, p, values, local_cache):
        runtime, channels, _ = self._path_inputs(p, values)
        output = runtime.evaluate(channels, _local_cache=local_cache)
        return self._path_output(p, output, values)

    def _path_output(self, p, output, values):
        from ye3t.core.tesseral import _real_to_complex_matrix

        path = self.path_programs[p]
        if self.path_basis != "real_tesseral":
            transform = _real_to_complex_matrix(path["output_L"], values.dtype, values.device)
            output = (((-1j)**path["real_phase_exponent"])*output @ transform.conj().T).real
        flattened = output.flatten(-2)
        if self.path_basis == "real_tesseral":
            binding = self._fixed_output_binding(f"path_output_map_{p}")
            if binding is not None:
                return flattened if binding[2] is None else flattened.index_select(-1, binding[2])
        return self._apply_output_map(p, flattened)

    def _path_batch_plan(self, values):
        """Bind shared physical-input and copy gathers from existing routes."""
        key = (values.dtype, values.device, self.path_basis)
        if key in self.path_batch_plans:
            return self.path_batch_plans[key]
        from ye3t.core.tesseral import _real_to_complex_matrix

        runtimes, local_specs, local_offsets, copy_offsets = {}, {}, {}, {}
        local_width = 0
        for p, path in self.path_programs.items():
            # Binding requires shape, dtype and device only. No physical values
            # or differentiable tensors enter the persistent execution plan.
            runtime, _, _ = self._path_inputs(p, values.new_empty((0, len(self.input_bindings[p]))))
            runtimes[p] = runtime
            channel_offsets, offset = [], 0
            for ell in path["input_Ls"]:
                channel_offsets.append(offset)
                offset += path["role_dimension"]*(2*ell+1)
            for bound, members in runtime.groups:
                row = runtime.schedule["local"][members[0][0]]
                channel = row["channel"]
                factor_width = 2*path["input_Ls"][channel]+1
                indices = tuple(self.input_bindings[p][channel_offsets[channel]+role*factor_width+m]
                    for role in bound.schedule["factor_roles"] for m in range(factor_width))
                if bound in local_specs:
                    if local_specs[bound][0] != indices:
                        raise ValueError("Shared local angular map has different physical input bindings")
                else:
                    transform = (None if self.path_basis == "real_tesseral" else
                                 _real_to_complex_matrix(bound.schedule["ell"], values.dtype, values.device))
                    local_specs[bound] = (indices, torch.tensor(indices, dtype=torch.long, device=values.device), transform)
                    local_offsets[bound] = local_width
                    local_width += bound.schedule["copies"]*bound.schedule["tableaux"]*(2*bound.schedule["output_L"]+1)
                for index, begin, end in members:
                    stride = bound.schedule["tableaux"]*(2*bound.schedule["output_L"]+1)
                    copy_offsets[p, index] = (local_offsets[bound]+begin*stride, stride, 2*bound.schedule["output_L"]+1)

        grouped, direct = {}, []
        for p, runtime in runtimes.items():
            route_groups = {}
            for route in runtime.schedule["routes"]:
                route_groups.setdefault(route["outer"], []).append(route)
            for index, ((tree, _), routes) in enumerate(zip(runtime.outer_groups, route_groups.values(), strict=True)):
                indices = []
                for route in routes:
                    for local, copy in route["local"]:
                        offset, stride, width = copy_offsets[p, local]
                        indices.extend(offset+copy*stride+m for m in range(width))
                count = len(routes)
                native = runtime.native_outer.get(id(tree))
                if native is None:
                    direct.append((p, index, torch.tensor(indices, dtype=torch.long, device=values.device), count))
                else:
                    if len(indices) != count*native[1]:
                        raise ValueError("Packed angular input indices disagree with the bound native width")
                    signature = runtime.native_outer_signatures[id(tree)]
                    group = grouped.setdefault(signature, dict(native=native, output_width=2*tree["L"]+1,
                        members=[], indices=[], offsets=[0]))
                    group["members"].append((p, index, count))
                    group["indices"].extend(indices)
                    group["offsets"].append(group["offsets"][-1]+count)
        for group in grouped.values():
            group["indices"] = torch.tensor(group["indices"], dtype=torch.long, device=values.device)
        plan = (runtimes, local_specs, tuple(grouped.values()), tuple(direct))
        self.path_batch_plans[key] = plan
        return plan

    def _local_path_values(self, values, local_specs):
        """Evaluate shared CG prefixes with exact physical-vector bindings.

        Group primary workspace is bounded by row count and scalar size. A
        single larger existing map remains intact. Backward retains additional
        arrays, so this is not a bound on whole-model training memory.
        """
        from .execution_plan import _PackedFactorizedAngularPlanGroup, _bounded_physical_factorized_groups

        rows = max(1, values.numel()//values.shape[-1])
        scalar_bytes = values.element_size()*(1 if self.path_basis == "real_tesseral" else 2)
        capacity = max(1, min(1024, _ANGULAR_BATCH_WORKSPACE_BYTES//(rows*scalar_bytes)))
        # Powers of two reuse bindings across nearby physical row counts.
        capacity = 1 << (capacity.bit_length()-1)
        key = values.dtype, values.device, self.path_basis, capacity
        if key not in self.local_group_plans:
            by_ell = {}
            for bound, (indices, _, transform) in local_specs.items():
                by_ell.setdefault(bound.schedule["ell"], []).append((bound, indices, transform))
            plans = []
            for ell, members in by_ell.items():
                width = 2*ell+1
                vectors, offsets, modules = {}, [], []
                for bound, indices, _ in members:
                    physical = []
                    for begin in range(0, len(indices), width):
                        vector = indices[begin:begin+width]
                        if len(vector) != width:
                            raise ValueError("Local angular binding cuts a magnetic multiplet")
                        if vector not in vectors:
                            vectors[vector] = len(vectors)*width
                        physical.append(vectors[vector])
                    offsets.append(tuple(physical))
                    modules.append(bound._native_plan)
                bank_indices = torch.tensor([value for vector in vectors for value in vector],
                                            dtype=torch.long, device=values.device)
                groups = []
                for indices in _bounded_physical_factorized_groups(modules, offsets, len(bank_indices), capacity):
                    group = _PackedFactorizedAngularPlanGroup([modules[index] for index in indices], indices,
                        physical_leaf_offsets=[offsets[index] for index in indices], physical_input_dimension=len(bank_indices),
                        heterogeneous_roots=True, share_physical_subtrees=True)
                    groups.append((group, [members[index] for index in indices]))
                plans.append((width, bank_indices, members[0][2], groups))
            self.local_group_plans[key] = plans
        outputs, dispatches, maximum_workspace = {}, 0, 0
        for width, indices, transform, groups in self.local_group_plans[key]:
            bank = values.index_select(-1, indices)
            if transform is not None:
                bank = (bank.reshape(*values.shape[:-1], len(indices)//width, width).to(transform.dtype)
                        @ transform).flatten(-2)
            for group, members in groups:
                packed = group._evaluate_packed_output(bank.reshape(-1, bank.shape[-1]).contiguous())
                parts = packed.split([record["stop"]-record["start"] for record in group.output_slices], dim=-1)
                for (bound, _, _), part in zip(members, parts, strict=True):
                    schedule = bound.schedule
                    dim = 2*schedule["output_L"]+1
                    paths = part.reshape(*values.shape[:-1], len(schedule["outputs"]), dim)
                    outputs[bound] = bound._mix_paths(paths).flatten(-2)
                dispatches += 1
                maximum_workspace = max(maximum_workspace, rows*scalar_bytes*group.workspace_dimension)
        self.last_local_group_report = dict(native_dispatches=dispatches,
            sequential_native_dispatches=len(local_specs), workspace_coordinate_capacity=capacity,
            max_primary_workspace_bytes=maximum_workspace)
        return [outputs[bound] for bound in local_specs]

    def _packed_path_output(self, values, plan, parts, placements, raw_width):
        """Compose existing fixed output maps; no coupling construction occurs.

        Raw grouped storage is restored to compiler route order, selected and
        scaled using the original maps. Bindings retain only immutable tensors
        and are bounded independently of the geometry or autograd graph.
        """
        runtimes = plan[0]
        bindings = {p: self._fixed_output_binding(f"path_output_map_{p}") for p in runtimes}
        buffers = [buffer for p in runtimes for buffer in
                   (getattr(self, f"outputs_{p}"), *self._output_buffers(p))]
        buffers.extend(runtime.outer_order for runtime in runtimes.values() if runtime.outer_order is not None)
        buffers.extend(binding[2] for binding in bindings.values()
                       if binding is not None and binding[2] is not None)
        factors = tuple(runtime.schedule["parent_factor"] for runtime in runtimes.values())
        key = (id(plan), factors, tuple(placements.items()),
               tuple((id(value), value._version) for value in buffers))
        if key not in self.packed_output_plans:
            indices, scales = [], []
            sparse_rows, sparse_columns, sparse_values, target_width = [], [], [], 0
            linear = any(binding is None for binding in bindings.values())
            for p, runtime in runtimes.items():
                width = 2*self.path_programs[p]["output_L"]+1
                ordered = []
                for index in range(len(runtime.outer_groups)):
                    start, count, actual_width = placements[p, index]
                    if actual_width != width:
                        raise ValueError("Packed output cuts an angular multiplet")
                    ordered.extend(range(start, start+count*width))
                if runtime.outer_order is not None:
                    ordered = [ordered[row*width+m] for row in runtime.outer_order.cpu().tolist() for m in range(width)]
                selected = getattr(self, f"outputs_{p}").cpu().tolist()
                if linear:
                    if f"path_output_map_{p}" in self._buffers:
                        coefficients = self._buffers[f"path_output_map_{p}"].detach().cpu()[:, selected]
                        entries = coefficients.nonzero()
                        triples = zip(entries[:, 0].tolist(), entries[:, 1].tolist(),
                                      (coefficients[entries[:, 0], entries[:, 1]]*
                                       runtime.schedule["parent_factor"]).tolist(), strict=True)
                    else:
                        rows, columns, coefficients = self._compact_output_tables(p)
                        selection = {}
                        for index, column in enumerate(selected):
                            selection.setdefault(column, []).append(index)
                        triples = sorted((row, index, value) for row, column, value in zip(
                            rows.cpu().tolist(), columns.cpu().tolist(),
                            (coefficients.detach().cpu()*runtime.schedule["parent_factor"]).tolist(), strict=True)
                            for index in selection.get(column, ()) if value != 0)
                    for row, column, value in triples:
                        sparse_rows.append(ordered[row])
                        sparse_columns.append(target_width+column)
                        sparse_values.append(value)
                    target_width += len(selected)
                else:
                    selection = bindings[p][2]
                    if selection is not None:
                        ordered = [ordered[index] for index in selection.cpu().tolist()]
                    indices.extend(ordered[index] for index in selected)
                    scales.extend([runtime.schedule["parent_factor"]]*len(selected))
            # Constants bound during inference must remain usable by autograd.
            with torch.inference_mode(False):
                if linear:
                    from .execution_plan import _load_extension
                    _load_extension()
                    identity = torch.arange(raw_width, device=values.device)
                    tables = (identity, identity, values.new_ones(raw_width),
                        torch.tensor(sparse_rows, device=values.device, dtype=torch.long),
                        torch.tensor(sparse_columns, device=values.device, dtype=torch.long),
                        values.new_tensor(sparse_values), raw_width, target_width)
                else:
                    tables = (torch.tensor(indices, device=values.device, dtype=torch.long),
                              values.new_tensor(scales))
            self.packed_output_plans[key] = (linear, tables, tuple(buffers), plan)
            self.packed_output_plan_builds += 1
            while len(self.packed_output_plans) > 4:
                self.packed_output_plans.popitem(last=False)
        self.packed_output_plans.move_to_end(key)
        linear, tables, _, _ = self.packed_output_plans[key]
        raw = packed_concatenate(parts) if parts else values[..., :0]
        parts.clear()
        self.packed_output_calls += 1
        if linear:
            self.packed_linear_output_calls += 1
            if tables[-1] == 0:
                return raw[..., :0]
            if raw_width == 0:
                return raw.sum(dim=-1, keepdim=True).expand(*values.shape[:-1], tables[-1])
            return torch.ops.ye3t_runtime.source_analysis(
                raw.reshape(-1, raw_width).contiguous(), *tables
            ).reshape(*values.shape[:-1], tables[-1])
        indices, scales = tables
        return raw.index_select(-1, indices)*scales

    def _path_forward_many(self, values, packed_output=False):
        """Execute shared local maps and batched outer maps with packed gathers.

        Compiler routes specify every copy and physical input. Persistent plans
        contain only fixed maps and indices; geometry remains local to this call.
        """
        plan = self._path_batch_plan(values)
        runtimes, local_specs, groups, direct = plan
        local = self._local_path_values(values, local_specs)
        packed_local = (packed_concatenate(local) if local else values.new_empty((*values.shape[:-1], 0)))
        del local
        outputs = {} if packed_output else {p: [None]*len(runtime.outer_groups) for p, runtime in runtimes.items()}
        remaining = {p: len(pieces) for p, pieces in outputs.items()}
        projected = {}
        parts, placements, raw_width = [], {}, 0
        result, output = None, None

        def finish(p):
            runtime = runtimes[p]
            pieces = outputs.pop(p)
            if not pieces:
                dtype = (values.dtype if self.path_basis == "real_tesseral" else
                         torch.complex128 if values.dtype == torch.float64 else torch.complex64)
                output = values.new_zeros((*values.shape[:-1], 0, 2*self.path_programs[p]["output_L"]+1), dtype=dtype)
            else:
                output = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)
                if runtime.outer_order is not None:
                    output = output.index_select(-2, runtime.outer_order)
                output = output*runtime.schedule["parent_factor"]
            projected[p] = self._path_output(p, output, values)

        for p, count in remaining.items():
            if count == 0:
                finish(p)
        for p, index, indices, count in direct:
            runtime = runtimes[p]
            width = 2*runtime.outer_groups[index][0]["L"]+1
            output = packed_local.index_select(-1, indices)
            if packed_output:
                parts.append(output)
                placements[p, index] = (raw_width, count, width)
                raw_width += count*width
            else:
                outputs[p][index] = output.reshape(*values.shape[:-1], count, width)
                remaining[p] -= 1
                if remaining[p] == 0:
                    finish(p)
        batch_rows = values.numel() // values.shape[-1]
        dispatches, max_workspace_bytes = 0, 0
        for group in groups:
            members = group["members"]
            arguments, width, workspace = group["native"]
            output_width = group["output_width"]
            element_bytes = arguments[-1].element_size()
            bytes_per_route = max(1, batch_rows)*(workspace+width+output_width)*element_bytes
            begin = 0
            while begin < len(members):
                spans, routes = [], 0
                end = begin
                while end < len(members):
                    p, index, count = members[end]
                    # Route counts are static shape metadata. Keep a large
                    # individual program's existing behavior, without adding
                    # other programs to its temporary workspace.
                    if spans and (routes+count)*bytes_per_route > _ANGULAR_BATCH_WORKSPACE_BYTES:
                        break
                    spans.append(count)
                    routes += count
                    end += 1
                indices = group["indices"][group["offsets"][begin]*width:group["offsets"][end]*width]
                result = _gathered_factorized_angular(
                    packed_local, indices, arguments, width, workspace, output_width
                ).reshape(*values.shape[:-1], routes, output_width)
                if packed_output:
                    parts.append(result.flatten(-2))
                    for p, index, count in members[begin:end]:
                        placements[p, index] = (raw_width, count, output_width)
                        raw_width += count*output_width
                else:
                    for (p, index, _), output in zip(members[begin:end], result.split(spans, dim=-2), strict=True):
                        outputs[p][index] = output
                        remaining[p] -= 1
                        if remaining[p] == 0:
                            finish(p)
                dispatches += 1
                max_workspace_bytes = max(max_workspace_bytes, routes*bytes_per_route)
                begin = end
        self.last_path_batch_report = dict(programs=len(runtimes), native_operator_groups=len(groups),
            native_dispatches=dispatches, sequential_native_dispatches=sum(len(group["members"]) for group in groups),
            local_input_gathers=len(local_specs), outer_input_gathers=dispatches+len(direct),
            max_estimated_workspace_bytes=max_workspace_bytes)
        if packed_output:
            # Native autograd retains any required values itself. Drop Python
            # owners before allocating the combined output, especially when
            # gradients are disabled and those intermediates are already dead.
            del packed_local, result, output
            return self._packed_path_output(values, plan, parts, placements, raw_width)
        return projected

    def _path_adjoint(self, p, values, seeds):
        from ye3t.core.tesseral import _real_to_complex_matrix

        path = self.path_programs[p]
        runtime, channels, forms = self._path_inputs(p, values)
        upstream = self._apply_output_map(p, seeds, transpose=True)
        upstream = upstream.reshape(*seeds.shape[:-1], len(path["routes"]), 2*path["output_L"]+1)
        if self.path_basis != "real_tesseral":
            transform = _real_to_complex_matrix(path["output_L"], values.dtype, values.device)
            upstream = ((-1j)**path["real_phase_exponent"])*upstream.to(transform.dtype) @ transform.conj()
        gradients = runtime.batched_adjoint(channels, upstream)
        if self.path_basis == "real_tesseral":
            return torch.cat([gradients[index].flatten(-2) for index in range(len(path["input_Ls"]))], dim=-1)
        return torch.cat([(gradients[index] @ forms[index].T).real.flatten(-2)
                          for index in range(len(path["input_Ls"]))], dim=-1)

    def _tables(self, p, s):
        return tuple(getattr(self, f"{name}_{p}_{s}") for name in (
            "counts", "output_offsets", "coefficient_terms", "coefficient_outputs", "coefficient_values"))

    def forward(self, values):
        if (self.backend == "native" and self.path_basis == "real_tesseral"
                and len(self.path_programs) > 1 and len(self.path_programs) == len(self.programs)
                and all(not buffer.requires_grad and not torch.is_inference(buffer)
                        for p in self.path_programs for buffer in
                        (*self._output_buffers(p), getattr(self, f"outputs_{p}")))):
            return self._path_forward_many(values, packed_output=True)
        outputs = []
        local_cache = {}
        paths = self._path_forward_many(values) if self.backend == "native" and len(self.path_programs) > 1 else {}
        for p, stages in enumerate(self.programs):
            if p in paths:
                outputs.append(self._select_outputs(p, paths[p]))
                continue
            current = values.index_select(-1, getattr(self, f"inputs_{p}"))
            if p in self.path_programs:
                outputs.append(self._select_outputs(p, self._path_forward(p, current, local_cache)))
                continue
            for s in range(len(stages)):
                current = symmetric_power_shared_monomial_contraction(current, *self._tables(p, s), backend=self.backend)
            outputs.append(self._select_outputs(p, current))
        # The shared direct-sum adjoint assembles mixed derivatives once instead
        # of scattering one full output-width gradient per producer block.
        return packed_concatenate(outputs)

    def batched_adjoint(self, values, seeds):
        """Explicit reverse product rule, without a seed-expanded input batch."""
        result = values.new_zeros((seeds.shape[0],)+tuple(values.shape))
        begin = 0
        for p, stages in enumerate(self.programs):
            input_indices = getattr(self, f"inputs_{p}")
            output_indices = getattr(self, f"outputs_{p}")
            if p in self.path_programs:
                count = len(output_indices)
                part = seeds[..., begin:begin+count]
                begin += count
                if (values.device.type == "cpu" and not values.requires_grad and not seeds.requires_grad
                        and not bool(torch.any(part != 0))):
                    # Exact zero design-row seeds need no reverse execution.
                    # Tracked model/force-loss derivatives always retain their graph.
                    continue
                adjoint = values.new_zeros((*seeds.shape[:-1], self.path_output_shapes[p][1]))
                adjoint = adjoint.index_add(-1, output_indices, part)
                result = result.index_add(-1, input_indices, self._path_adjoint(
                    p, values.index_select(-1, input_indices), adjoint))
                continue
            states = [values.index_select(-1, input_indices)]
            for s in range(len(stages)-1):
                states.append(symmetric_power_shared_monomial_contraction(
                    states[-1], *self._tables(p, s), backend=self.backend))
            adjoint = values.new_zeros((seeds.shape[0], values.shape[0], stages[-1]["output_dimension"]))
            adjoint.index_add_(-1, output_indices, seeds[..., begin:begin+len(output_indices)])
            begin += len(output_indices)
            for s in reversed(range(len(stages))):
                adjoint = symmetric_power_shared_monomial_batched_adjoint(
                    adjoint, states[s], *self._tables(p, s), backend=self.backend)
            result.index_add_(-1, input_indices, adjoint)
        return result
