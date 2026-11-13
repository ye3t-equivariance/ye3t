"""Source-independent execution of compiler-owned Young/angular path maps."""
from functools import lru_cache
from types import SimpleNamespace


@lru_cache(maxsize=512)
def _real_path_entries(angular, entries):
    """Verify the saved canonical map and reuse its existing exact real form."""
    from ye3t.core.cg import _cg_table_exact
    from ye3t.core.tesseral import _real_cg_entries_exact

    a, b, c = angular
    canonical = tuple((m+a, n+b, k+c, float(value)) for m, n, k, value in _cg_table_exact(a, b, c))
    if sorted(entries) != sorted(canonical):
        raise ValueError("Stored CG coefficients differ from the canonical normalized map")
    return tuple((i, j, k, float(value)) for i, j, k, value in _real_cg_entries_exact(a, b, c))


def _bound_cg_table(angular, entries, dtype, device, real):
    """Bind supplied compiler coefficients in the existing real/complex basis.

    The real form is the unitary coordinate change from core.tesseral, with
    the documented -i phase for odd a+b-c. No coupling labels are discovered.
    Conversion reuses the exact core map; retained tables are real in real mode.
    """
    import torch

    shape = tuple(2*L+1 for L in angular)
    if not real:
        table = torch.zeros(shape, dtype=dtype, device=device)
        for a, b, c, value in entries:
            table[a, b, c] = value
        return table, entries
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("Real angular storage requires float32 or float64")
    bound_entries = _real_path_entries(tuple(angular), tuple(map(tuple, entries)))
    table = torch.zeros(shape, dtype=dtype)
    for a, b, c, value in bound_entries:
        table[a, b, c] = value
    return table.to(dtype=dtype, device=device), bound_entries


def _real_merge_phase(left_phase, right_phase, angular):
    """Compose existing real forms, preserving the source's overall phase.

    If each child has phase (-i)^p and the binary CG has phase (-i)^b,
    the parent's phase exponent is (p_left+p_right+b) mod2. The integer
    quotient supplies the compensating sign. This is coordinate bookkeeping.
    """
    total = left_phase+right_phase+(angular[0]+angular[1]-angular[2]) % 2
    return total % 2, -1 if total//2 % 2 else 1


def _validate_angular_schedule(schedule):
    """Check stored topology and complete axes once, without rebuilding maps."""
    from math import isfinite

    if schedule["kind"] != "young_angular_paths_v1":
        raise ValueError("Unsupported angular-path schedule")
    size, ell, output = schedule["size"], schedule["ell"], schedule["output_L"]
    if size < 1 or min(ell, output, schedule["copies"], schedule["tableaux"]) < 0:
        raise ValueError("Invalid angular-path dimensions")
    tables = schedule["cg_tables"]
    for table in tables:
        a, b, c = table["angular_momenta"]
        if min(a, b, c) < 0 or not abs(a-b) <= c <= a+b:
            raise ValueError("Invalid angular-path CG triangle")
        for m, n, k, value in table["entries"]:
            if (not (0 <= m < 2*a+1 and 0 <= n < 2*b+1 and 0 <= k < 2*c+1)
                    or m-a+n-b != k-c or not isfinite(value)):
                raise ValueError("Invalid angular-path CG entry")
    angular, depth = [ell], [0]
    for index, (previous, factor, table) in enumerate(schedule["nodes"], 1):
        if not (0 <= previous < index and 0 < factor < size and 0 <= table < len(tables)):
            raise ValueError("Invalid angular-path dependency")
        a, b, c = tables[table]["angular_momenta"]
        if a != angular[previous] or b != ell or factor != depth[previous]+1:
            raise ValueError("Angular-path factors or CG dimensions do not compose")
        angular.append(c)
        depth.append(factor)
    for node in schedule["outputs"]:
        if not (0 <= node < len(angular)) or angular[node] != output or depth[node] != size-1:
            raise ValueError("Angular-path output is not a complete target multiplet")
    for path, column, value in schedule["mixing"]:
        if (not 0 <= path < len(schedule["outputs"])
                or not 0 <= column < schedule["copies"]*schedule["tableaux"] or not isfinite(value)):
            raise ValueError("Invalid angular-path multiplicity map")


class BoundAngularPaths:
    """Bind immutable CG tables and multiplicity mixing once per device/dtype."""

    def __init__(self, schedule, *, dtype, device, backend="reference", basis="complex_condon_shortley"):
        import torch

        _validate_angular_schedule(schedule)
        if backend not in {"reference", "native"}:
            raise ValueError("Angular-path execution requires reference or native")
        self.schedule = schedule
        self.backend = backend
        if basis not in {"complex_condon_shortley", "real_tesseral"}:
            raise ValueError("Unknown angular-path basis")
        if basis == "real_tesseral" and dtype not in {torch.float32, torch.float64}:
            raise ValueError("Real angular storage requires float32 or float64")
        self.basis = basis
        self.native_arguments = None
        self.tables = []
        self.table_entries = []
        for table in schedule["cg_tables"]:
            angular = table["angular_momenta"]
            values, entries = _bound_cg_table(angular, table["entries"], dtype, device, basis == "real_tesseral")
            self.tables.append(values)
            self.table_entries.append(entries)
        phases, self.node_signs = [0], []
        for previous, factor, table in schedule["nodes"]:
            phase, sign = _real_merge_phase(phases[previous], 0, schedule["cg_tables"][table]["angular_momenta"])
            phases.append(phase)
            self.node_signs.append(sign if basis == "real_tesseral" else 1)
        self.paths = torch.tensor([p for p, _, _ in schedule["mixing"]],
                                  dtype=torch.long, device=device)
        self.columns = torch.tensor([c for _, c, _ in schedule["mixing"]],
                                    dtype=torch.long, device=device)
        self.coefficients = torch.tensor([v for _, _, v in schedule["mixing"]],
                                         dtype=dtype, device=device)
        self._mixing_buffers = (self.paths, self.columns, self.coefficients)
        self._mixing_versions = (None if any(torch.is_inference(value) for value in self._mixing_buffers)
                                 else tuple(value._version for value in self._mixing_buffers))
        self._mixing_dimensions = (len(schedule["outputs"]), schedule["copies"]*schedule["tableaux"],
                                   2*schedule["output_L"]+1)
        self._mixing_identity = None
        # Nodes with the same added factor and CG map can execute together.
        # Every dependency has one fewer factors, so grouping by depth is safe.
        grouped = {}
        for index, (previous, factor, table) in enumerate(schedule["nodes"], 1):
            grouped.setdefault((factor, table, self.node_signs[index-1]), []).append((index, previous))
        self.groups = tuple((factor, table, sign, tuple(rows))
                            for (factor, table, sign), rows in sorted(grouped.items()))
        if backend == "native" and schedule["outputs"]:
            self._bind_native(dtype, device)

    def _bind_native(self, dtype, device):
        """Bind the existing shared native angular-DAG kernel, without new maps."""
        import torch
        from .execution_plan import _load_extension

        extension = _load_extension()
        if torch.device(device).type == "cuda" and not extension.has_cuda():
            raise RuntimeError("Installed YE3T native angular kernel has no CUDA support")
        schedule = self.schedule
        size, dim = schedule["size"], 2*schedule["ell"]+1
        dimensions, leaves = [dim]*size, [f*dim for f in range(size)]
        left, right, coefficient_offsets = [-1]*size, [-1]*size, [0]*(size+1)
        rows, columns, coefficients = [], [], []
        translation = {0: 0}
        for node, (previous, factor, table_index) in enumerate(schedule["nodes"], 1):
            table = schedule["cg_tables"][table_index]
            translation[node] = len(dimensions)
            dimensions.append(2*table["angular_momenta"][2]+1)
            leaves.append(-1)
            left.append(translation[previous])
            right.append(factor)
            for a, b, c, value in self.table_entries[table_index]:
                rows.append(a*dim+b)
                columns.append(c)
                coefficients.append(value*self.node_signs[node-1])
            coefficient_offsets.append(len(rows))
        offsets, width = [], 0
        for dimension in dimensions:
            offsets.append(width)
            width += dimension
        integers = (offsets, dimensions, leaves, left, right, coefficient_offsets, rows, columns)
        self.native_arguments = tuple(torch.tensor(x, dtype=torch.int64, device=device) for x in integers) + (
            torch.tensor(coefficients, dtype=dtype, device=device),
            torch.tensor([translation[node] for node in schedule["outputs"]], dtype=torch.int64, device=device),
            torch.ones(1, dtype=dtype, device=device))
        self.workspace_dimension = width
        # Adapt the already bound CG DAG to the shared heterogeneous executor.
        # Exact operator keys include the coordinate convention and phase;
        # physical leaves are supplied by its caller. No new maps are built.
        operators = [("leaf", self.basis, schedule["ell"])]*size
        for node, (_, _, table) in enumerate(schedule["nodes"]):
            operators.append(("merge", self.basis,
                tuple(schedule["cg_tables"][table]["angular_momenta"]),
                self.node_signs[node], tuple(map(tuple, self.table_entries[table]))))
        names = ("node_offsets", "node_dimensions", "node_leaf_offsets", "node_left",
                 "node_right", "node_coefficient_offsets", "coefficient_rows",
                 "coefficient_columns", "coefficient_values", "root_nodes", "projection_values")
        magnetic = 2*schedule["output_L"]+1
        self._native_plan = SimpleNamespace(
            **dict(zip(names, self.native_arguments, strict=True)),
            node_operator_keys=tuple(operators), input_Ls=(schedule["ell"],)*size,
            workspace_dimension=width, resolved_backend="native", source_dimension=1,
            magnetic_dimension=magnetic, input_dimension=size*dim,
            output_dimension=len(schedule["outputs"])*magnetic,
            logical_channel_tableau_dimension=len(schedule["outputs"]),
            compiler_schedule=schedule)

    def _native_paths(self, factors):
        import torch

        packed = torch.cat(factors, dim=-1)
        shape = packed.shape[:-1]
        output_width = len(self.schedule["outputs"])*(2*self.schedule["output_L"]+1)
        result = torch.ops.ye3t_runtime.factorized_angular(
            packed.reshape(-1, packed.shape[-1]).contiguous(), *self.native_arguments,
            1, self.workspace_dimension, output_width)
        return result.reshape(*shape, len(self.schedule["outputs"]), 2*self.schedule["output_L"]+1)

    def _states(self, factors):
        import torch

        nodes = [factors[0]] + [None]*len(self.schedule["nodes"])
        for factor, table, sign, rows in self.groups:
            left = torch.stack([nodes[previous] for _, previous in rows], dim=-2)
            output = sign*torch.einsum("...pm,...n,mnk->...pk", left, factors[factor], self.tables[table])
            for index, (node, _) in enumerate(rows):
                nodes[node] = output[..., index, :]
        return nodes

    def _mix_paths(self, values, transpose=False):
        """Apply the existing multiplicity map with complete magnetic axes.

        Native fixed-map application treats M as another batch axis and avoids
        a physical-row by nonzero-entry temporary. The algebraic transpose uses
        the same conjugated coefficients as the analytical covector path.
        Edited or differentiable coefficient tables retain the reference path.
        """
        import torch

        schedule = self.schedule
        paths, copies, width = (len(schedule["outputs"]), schedule["copies"]*schedule["tableaux"],
                                2*schedule["output_L"]+1)
        input_width, output_width = (copies, paths) if transpose else (paths, copies)
        if values.shape[-2:] != (input_width, width):
            raise ValueError("Multiplicity mixing requires complete path/copy and magnetic axes")
        buffers = (self.paths, self.columns, self.coefficients)
        fixed = (self._mixing_versions is not None
                 and all(current is original for current, original in zip(buffers, self._mixing_buffers))
                 and not any(torch.is_inference(value) for value in buffers)
                 and tuple(value._version for value in buffers) == self._mixing_versions
                 and (paths, copies, width) == self._mixing_dimensions)
        if (self.backend == "native" and fixed and not self.coefficients.requires_grad
                and paths > 0 and copies > 0 and values.dtype == self.coefficients.dtype
                and values.device == self.coefficients.device):
            if self._mixing_identity is None:
                # Inference warm-up may precede differentiable evaluation.
                with torch.inference_mode(False):
                    self._mixing_identity = (torch.arange(paths, device=values.device),
                                            torch.ones(paths, dtype=values.dtype, device=values.device))
            identity, ones = self._mixing_identity
            arguments = (identity, identity, ones, self.paths, self.columns, self.coefficients)
            flat = (values.conj() if transpose else values).transpose(-1, -2).reshape(-1, input_width).contiguous()
            if transpose:
                output = torch.ops.ye3t_runtime.source_analysis_adjoint(flat, *arguments, paths, paths).conj()
            else:
                output = torch.ops.ye3t_runtime.source_analysis(flat, *arguments, paths, copies)
            return output.reshape(*values.shape[:-2], width, output_width).transpose(-1, -2)
        rows, columns = (self.columns, self.paths) if transpose else (self.paths, self.columns)
        contribution = values[..., rows, :]*self.coefficients.conj().unsqueeze(-1)
        output = values.new_zeros((*values.shape[:-2], output_width, width))
        return output.index_add(-2, columns, contribution)

    def evaluate(self, factors):
        import torch

        schedule = self.schedule
        if len(factors) != schedule["size"]:
            raise ValueError("Factor count differs from angular-path schedule")
        shape = factors[0].shape[:-1]
        if any(value.shape != (*shape, 2*schedule["ell"]+1) for value in factors):
            raise ValueError("Angular factors require matching batches and complete multiplets")
        width = 2*schedule["output_L"]+1
        result = factors[0].new_zeros((*shape, schedule["copies"]*schedule["tableaux"], width))
        if schedule["outputs"]:
            if self.native_arguments is not None:
                paths = self._native_paths(factors)
            else:
                nodes = self._states(factors)
                paths = torch.stack([nodes[index] for index in schedule["outputs"]], dim=-2)
            result = self._mix_paths(paths)
        return result.reshape(*shape, schedule["copies"], schedule["tableaux"], width)

    def batched_adjoint(self, factors, seeds):
        """Algebraic transpose Jacobian with an initial seed axis.

        Explicit binary product rules preserve differentiation through the
        resulting force rows. No autograd Jacobian or magnetic tensor is built.
        Complex callers supply seeds in the algebraic covector convention.
        """
        import torch

        schedule = self.schedule
        expected = (*factors[0].shape[:-1], schedule["copies"], schedule["tableaux"],
                    2*schedule["output_L"]+1)
        if tuple(seeds.shape[1:]) != expected:
            raise ValueError("Angular-path adjoint seeds require complete output axes")
        if self.native_arguments is not None:
            # Bound repeated native workspace across adjoint seeds. The existing
            # kernel retains its double backward, including complex covectors.
            batch = factors[0].numel()//factors[0].shape[-1]
            per_seed = max(1, batch*self.workspace_dimension*factors[0].element_size()*4)
            chunk = max(1, (32*1024**2)//per_seed)
            if seeds.shape[0] > chunk:
                pieces = [self.batched_adjoint(factors, seeds[first:first+chunk])
                          for first in range(0,seeds.shape[0],chunk)]
                return tuple(torch.cat([piece[i] for piece in pieces], dim=0) for i in range(len(factors)))
        width = 2*schedule["output_L"]+1
        # Keep compiler-owned dimensions explicit: -1 cannot be inferred when
        # a physical batch or the batch of derivative seeds is empty.
        seeds = seeds.reshape(seeds.shape[0], *factors[0].shape[:-1],
                              schedule["copies"]*schedule["tableaux"], width)
        path_seeds = self._mix_paths(seeds, transpose=True)
        if self.native_arguments is not None:
            packed = torch.cat(factors, dim=-1)
            repeated = packed.unsqueeze(0).expand(seeds.shape[0], *packed.shape).reshape(-1,packed.shape[-1])
            gradient = torch.ops.ye3t_runtime.factorized_angular_adjoint(
                path_seeds.conj().reshape(repeated.shape[0], len(schedule["outputs"])*width).contiguous(),
                repeated.contiguous(),
                *self.native_arguments, 1, self.workspace_dimension).conj()
            gradient = gradient.reshape(seeds.shape[0], *packed.shape)
            return tuple(gradient.split(2*schedule["ell"]+1, dim=-1))
        nodes = self._states(factors)
        adjoints = [None]*len(nodes)
        for index, node in enumerate(schedule["outputs"]):
            adjoints[node] = path_seeds[..., index, :]
        factor_adjoints = [seeds.new_zeros((seeds.shape[0],)+tuple(factor.shape)) for factor in factors]
        for factor, table, sign, rows in reversed(self.groups):
            pending = [(node, previous) for node, previous in rows if adjoints[node] is not None]
            if not pending:
                continue
            upstream = sign*torch.stack([adjoints[node] for node, _ in pending], dim=-2)
            left = torch.stack([nodes[previous] for _, previous in pending], dim=-2)
            left_adjoint = torch.einsum("s...pk,...n,mnk->s...pm", upstream, factors[factor], self.tables[table])
            factor_adjoints[factor] = factor_adjoints[factor] + torch.einsum(
                "s...pk,...pm,mnk->s...n", upstream, left, self.tables[table])
            for index, (_, previous) in enumerate(pending):
                value = left_adjoint[..., index, :]
                adjoints[previous] = value if adjoints[previous] is None else adjoints[previous]+value
        if adjoints[0] is not None:
            factor_adjoints[0] = factor_adjoints[0]+adjoints[0]
        return tuple(factor_adjoints)


def _pack_angular_schedules(schedules):
    """Share identical CG prefixes across compiler-owned multiplicity maps.

    This is an execution layout. Original sector labels and copy coordinates
    stay in the compiler schedule; packing only concatenates their output axes.
    """
    first = schedules[0]
    tables, table_index, nodes, node_index, roots, root_index, mixing = [], {}, [], {}, [], {}, []
    ranges, copies = [], 0
    for schedule in schedules:
        _validate_angular_schedule(schedule)
        if any(schedule[key] != first[key] for key in ("size","ell","output_L","factor_roles")):
            raise ValueError("Packed angular maps need identical factor bindings and output L")
        local_tables = []
        for table in schedule["cg_tables"]:
            key = tuple(table["angular_momenta"])
            if key not in table_index:
                table_index[key] = len(tables)
                tables.append(table)
            elif tuple(map(tuple,tables[table_index[key]]["entries"])) != tuple(map(tuple,table["entries"])):
                raise ValueError("Compatible CG prefixes have inconsistent coefficient maps")
            local_tables.append(table_index[key])
        translate = {0:0}
        for index,(previous,factor,table) in enumerate(schedule["nodes"],1):
            key = (translate[previous],factor,local_tables[table])
            if key not in node_index:
                node_index[key] = len(nodes)+1
                nodes.append(key)
            translate[index] = node_index[key]
        output_indices = []
        for node in schedule["outputs"]:
            global_node = translate[node]
            if global_node not in root_index:
                root_index[global_node] = len(roots)
                roots.append(global_node)
            output_indices.append(root_index[global_node])
        count = schedule["copies"]*schedule["tableaux"]
        mixing.extend((output_indices[p],copies+c,value) for p,c,value in schedule["mixing"])
        ranges.append((copies,copies+count))
        copies += count
    packed = {key:first[key] for key in ("kind","size","ell","output_L","factor_roles")}
    packed.update(copies=copies,tableaux=1,nodes=tuple(nodes),cg_tables=tuple(tables),
                  outputs=tuple(roots),mixing=tuple(mixing),copy_gauge="packed_compiler_owned_maps")
    return packed,tuple(ranges)


class BoundCauchyPaths:
    """Persistent shared path/CG execution for role-valued channel multiplets."""

    def __init__(self, schedule, *, dtype, device, backend="reference", basis="complex_condon_shortley",
                 _local_pool=None, _channel_bindings=None):
        import torch
        from ye3t.couplings.factorized_typed import _bind_cg_trees

        if schedule["kind"] != "normalized_cauchy_paths_v1":
            raise ValueError("Unsupported Cauchy path schedule")
        if basis == "real_tesseral" and dtype not in {torch.float32, torch.float64}:
            raise ValueError("Real Cauchy storage requires float32 or float64")
        self.schedule = schedule
        self.basis = basis
        self.native_outer = {}
        self.native_outer_signatures = {}
        self.cg_entries = {}
        grouped = {}
        for i,row in enumerate(schedule["local"]):
            local = row["schedule"]
            key = row["channel"],tuple(local["factor_roles"]),local["ell"],local["output_L"]
            grouped.setdefault(key,[]).append(i)
        self.local = [None]*len(schedule["local"])
        self.groups = []
        for members in grouped.values():
            packed,ranges = _pack_angular_schedules([schedule["local"][i]["schedule"] for i in members])
            # Equal representation labels alone do not establish equal maps or
            # physical inputs. Reuse requires both full schedule equality and
            # the exact input component sequence supplied by the parent plan.
            candidates = []
            if _local_pool is not None:
                channel = schedule["local"][members[0]]["channel"]
                key = (_channel_bindings[channel], dtype, str(device), backend, basis,
                       packed["ell"], packed["output_L"], tuple(packed["factor_roles"]))
                candidates = _local_pool.setdefault(key, [])
            runtime = next((item for item in candidates if item.schedule == packed), None)
            if runtime is None:
                runtime = BoundAngularPaths(packed,dtype=dtype,device=device,backend=backend,basis=basis)
                if _local_pool is not None:
                    candidates.append(runtime)
            self.groups.append((runtime,tuple((i,*span) for i,span in zip(members,ranges))))
            for i in members:
                self.local[i] = runtime
        for row in schedule["local"]:
            channel, local = row["channel"], row["schedule"]
            if (not 0 <= channel < len(schedule["input_Ls"])
                    or local["ell"] != schedule["input_Ls"][channel]
                    or len(local["factor_roles"]) != local["size"]
                    or any(not 0 <= role < schedule["role_dimension"] for role in local["factor_roles"])):
                raise ValueError("Invalid Cauchy role/channel binding")
        for route in schedule["routes"]:
            if not 0 <= route["outer"] < len(schedule["outer"]):
                raise ValueError("Invalid Cauchy outer route")
            for local, copy in route["local"]:
                if not 0 <= local < len(self.local) or not 0 <= copy < schedule["local"][local]["schedule"]["copies"]:
                    raise ValueError("Invalid Cauchy multiplicity binding")
        if basis == "complex_condon_shortley":
            self.cg = _bind_cg_trees(schedule["outer"], dtype, device)
        elif basis == "real_tesseral":
            self.cg = {}
            bound = {}
            for route in schedule["routes"]:
                phases = tuple((schedule["local"][i]["schedule"]["size"]*schedule["local"][i]["schedule"]["ell"]
                                -schedule["local"][i]["schedule"]["output_L"]) % 2 for i, copy in route["local"])
                phase = self._bind_real_tree(schedule["outer"][route["outer"]], phases, dtype, device, bound)
                if phase != schedule["real_phase_exponent"]:
                    raise ValueError("Cauchy routes disagree with the saved real-form phase")
        else:
            raise ValueError("Unknown Cauchy path basis")
        self._bind_outer_routes(device)
        if backend == "native":
            self._bind_outer_native(dtype, device)

    def _bind_outer_routes(self, device):
        """Bind copy gathers for routes sharing an existing compiler CG tree."""
        import torch

        def restore(order):
            if order == list(range(len(order))):
                return None
            return torch.argsort(torch.tensor(order, dtype=torch.long, device=device))

        grouped = {}
        for index, route in enumerate(self.schedule["routes"]):
            grouped.setdefault(route["outer"], []).append((index, route["local"]))
        self.outer_groups, order = [], []
        for outer, members in grouped.items():
            leaves = []
            for leaf in range(len(members[0][1])):
                local_groups = {}
                for position, (_, inputs) in enumerate(members):
                    local, copy = inputs[leaf]
                    local_groups.setdefault(local, []).append((position, copy))
                gathers, leaf_order = [], []
                for local, copies in local_groups.items():
                    leaf_order.extend(position for position, copy in copies)
                    indices = tuple(copy for position, copy in copies)
                    # Contiguous copies need only a view, including singleton
                    # routes. Arbitrary/repeated copies use the same gather.
                    selection = (slice(indices[0], indices[-1]+1)
                                 if indices == tuple(range(indices[0], indices[0]+len(indices)))
                                 else torch.tensor(indices, dtype=torch.long, device=device))
                    gathers.append((local, selection))
                leaves.append((tuple(gathers), restore(leaf_order)))
            self.outer_groups.append((self.schedule["outer"][outer], tuple(leaves)))
            order.extend(index for index, inputs in members)
        self.outer_order = restore(order)

    def _outer_inputs(self, local, leaves):
        import torch

        inputs = []
        for gathers, order in leaves:
            pieces = [(local[index][..., selection, 0, :] if isinstance(selection, slice)
                       else local[index][..., 0, :].index_select(-2, selection))
                      for index, selection in gathers]
            value = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)
            inputs.append(value if order is None else value.index_select(-2, order))
        return inputs

    def _bind_real_tree(self, tree, phases, dtype, device, bound):
        if tree["kind"] == "leaf":
            return phases[tree["index"]]
        key = (id(tree), phases)
        if key in bound:
            return bound[key]
        left = self._bind_real_tree(tree["left"], phases, dtype, device, bound)
        right = self._bind_real_tree(tree["right"], phases, dtype, device, bound)
        angular = (tree["left"]["L"], tree["right"]["L"], tree["L"])
        phase, sign = _real_merge_phase(left, right, angular)
        table, entries = _bound_cg_table(angular, tree["entries"], dtype, device, True)
        table = sign*table
        if id(tree) in self.cg:
            import torch
            if not torch.equal(self.cg[id(tree)], table):
                raise ValueError("Shared Cauchy tree has incompatible real-form bindings")
        else:
            self.cg[id(tree)] = table
            self.cg_entries[id(tree)] = tuple((a, b, c, sign*value) for a, b, c, value in entries)
        bound[key] = phase
        return phase

    def _bind_outer_native(self, dtype, device):
        """Pack existing compiler trees into the shared native angular DAG ABI.

        Leaf dimensions may differ. Coefficients and real-form phase signs are
        exactly those used by _tree_values; this only prepares execution indices.
        """
        import torch
        from .execution_plan import _load_extension

        extension = _load_extension()
        if torch.device(device).type == "cuda" and not extension.has_cuda():
            raise RuntimeError("Installed YE3T native angular kernel has no CUDA support")
        for tree, leaves in self.outer_groups:
            if tree["kind"] == "leaf":
                continue
            leaf_dimensions = {}
            stack = [tree]
            while stack:
                node = stack.pop()
                if node["kind"] == "leaf":
                    dimension = 2*node["L"]+1
                    if node["index"] in leaf_dimensions and leaf_dimensions[node["index"]] != dimension:
                        raise ValueError("Repeated CG leaf has inconsistent angular dimensions")
                    leaf_dimensions[node["index"]] = dimension
                else:
                    stack.extend((node["left"], node["right"]))
            if set(leaf_dimensions) != set(range(len(leaves))):
                raise ValueError("Outer CG tree does not bind every input multiplet")
            dimensions = [leaf_dimensions[i] for i in range(len(leaves))]
            input_offsets, width = [], 0
            for dimension in dimensions:
                input_offsets.append(width)
                width += dimension
            left, right = [-1]*len(leaves), [-1]*len(leaves)
            coefficient_offsets = [0]*(len(leaves)+1)
            rows, columns, coefficients, translated = [], [], [], {}
            def visit(node):
                if node["kind"] == "leaf":
                    return node["index"]
                if id(node) in translated:
                    return translated[id(node)]
                a, b = visit(node["left"]), visit(node["right"])
                index = len(dimensions)
                dimensions.append(2*node["L"]+1)
                input_offsets.append(-1)
                left.append(a)
                right.append(b)
                entries = self.cg_entries[id(node)] if self.basis == "real_tesseral" else node["entries"]
                for i, j, k, value in entries:
                    rows.append(i*dimensions[b]+j)
                    columns.append(k)
                    coefficients.append(value)
                coefficient_offsets.append(len(rows))
                translated[id(node)] = index
                return index
            root = visit(tree)
            offsets, workspace = [], 0
            for dimension in dimensions:
                offsets.append(workspace)
                workspace += dimension
            integers = (offsets, dimensions, input_offsets, left, right, coefficient_offsets, rows, columns)
            arguments = tuple(torch.tensor(value, dtype=torch.int64, device=device) for value in integers) + (
                torch.tensor(coefficients, dtype=dtype, device=device),
                torch.tensor([root], dtype=torch.int64, device=device), torch.ones(1, dtype=dtype, device=device))
            self.native_outer[id(tree)] = (arguments, width, workspace)
            # Exact bound tables identify compatible execution, including real
            # phase signs. Build this once from CPU schedule data; no tensor
            # hashing or device synchronization belongs in repeated evaluation.
            self.native_outer_signatures[id(tree)] = (
                tuple(tuple(value) for value in integers), tuple(coefficients), root, tree["L"])

    def _local_values(self, channels, needed=None, cache=None):
        result = [None]*len(self.local)
        for runtime,members in self.groups:
            if needed is not None and not any(i in needed for i,begin,end in members):
                continue
            row = self.schedule["local"][members[0][0]]
            values = cache.get(runtime) if cache is not None else None
            if values is None:
                values = runtime.evaluate([channels[row["channel"]][..., role, :]
                    for role in row["schedule"]["factor_roles"]])
                if cache is not None:
                    cache[runtime] = values
            for i,begin,end in members:
                result[i] = values[...,begin:end,:,:]
        return tuple(result)

    def _tree_values(self, tree, inputs, states):
        import torch

        if id(tree) not in states:
            if tree["kind"] == "leaf":
                states[id(tree)] = inputs[tree["index"]]
            else:
                states[id(tree)] = torch.einsum("...m,...n,mnk->...k",
                    self._tree_values(tree["left"], inputs, states),
                    self._tree_values(tree["right"], inputs, states), self.cg[id(tree)])
        return states[id(tree)]

    def evaluate(self, channels, *, _local_cache=None):
        import torch

        local = self._local_values(channels, cache=_local_cache)
        outputs = []
        for tree, leaves in self.outer_groups:
            inputs = self._outer_inputs(local, leaves)
            native = self.native_outer.get(id(tree))
            if native is None:
                output = self._tree_values(tree, inputs, {})
            else:
                arguments, input_width, workspace = native
                packed = torch.cat(inputs, dim=-1)
                output_width = 2*tree["L"]+1
                output = torch.ops.ye3t_runtime.factorized_angular(
                    packed.reshape(-1, input_width).contiguous(), *arguments, 1, workspace, output_width
                ).reshape(*packed.shape[:-1], output_width)
            outputs.append(output)
        if not outputs:
            first = channels[0]
            return first.new_zeros((*first.shape[:-2], 0, 2*self.schedule["output_L"]+1))
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-2)
        if self.outer_order is not None:
            output = output.index_select(-2, self.outer_order)
        return output*self.schedule["parent_factor"]

    def batched_adjoint(self, channels, seeds):
        """Explicit role, local-path and outer-CG product-rule transpose."""
        import torch

        static_rows = (seeds.device.type == "cpu" and not seeds.requires_grad
                       and not any(value.requires_grad for value in channels.values()))
        active = [index for index in range(len(self.schedule["routes"]))
                  if not static_rows or bool(torch.any(seeds[..., index, :] != 0))]
        needed = {i for index in active for i,copy in self.schedule["routes"][index]["local"]}
        local = self._local_values(channels,needed)
        local_seeds = [seeds.new_zeros((seeds.shape[0],)+tuple(value.shape)) if value is not None else None
                       for value in local]
        for index, route in enumerate(self.schedule["routes"]):
            if index not in active:
                continue
            inputs = [local[i][..., copy, 0, :] for i, copy in route["local"]]
            tree = self.schedule["outer"][route["outer"]]
            states = {}
            self._tree_values(tree, inputs, states)

            def reverse(node, upstream):
                if node["kind"] == "leaf":
                    i, copy = route["local"][node["index"]]
                    destination = torch.tensor([copy], dtype=torch.long, device=upstream.device)
                    local_seeds[i] = local_seeds[i].index_add(
                        -3, destination, upstream.unsqueeze(-2).unsqueeze(-2))
                else:
                    left, right, table = node["left"], node["right"], self.cg[id(node)]
                    reverse(left, torch.einsum("s...k,...n,mnk->s...m", upstream, states[id(right)], table))
                    reverse(right, torch.einsum("s...k,...m,mnk->s...n", upstream, states[id(left)], table))

            reverse(tree, seeds[..., index, :]*self.schedule["parent_factor"])
        result = {i: seeds.new_zeros((seeds.shape[0],)+tuple(value.shape)) for i, value in channels.items()}
        for runtime,members in self.groups:
            if local_seeds[members[0][0]] is None:
                continue
            row = self.schedule["local"][members[0][0]]
            upstream = torch.cat([local_seeds[i] for i,begin,end in members],dim=-3)
            channel = row["channel"]
            roles = row["schedule"]["factor_roles"]
            gradients = runtime.batched_adjoint([channels[channel][..., role, :] for role in roles], upstream)
            indices = torch.tensor(roles, dtype=torch.long, device=seeds.device)
            result[channel] = result[channel].index_add(-2, indices, torch.stack(gradients, dim=-2))
        return result
