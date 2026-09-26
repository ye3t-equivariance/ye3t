"""Runtime for compiler-owned hierarchical repeated-block instructions."""

import json
import math

import torch

from ye3t.couplings import SymmetricPowerProductPlan
from ye3t.execution_plan import YE3TExecutionPlan
from ye3t.runtime.symmetric_power import (
    _shared_symmetric_power_table,
    _symmetric_power_product_plan_count_table,
    symmetric_power_product_plan_contraction,
    symmetric_power_product_plan_table_report,
)
from ye3t.runtime.execution_plan import (
    _extension_for_native_dispatch,
    symmetric_power_shared_monomial_contraction,
)


def _dense_synthesis_table(table, dtype, device):
    matrix = torch.zeros(
        (int(table.input_dimension), int(table.output_dimension)),
        dtype=dtype,
        device=device,
    )
    if table.values:
        rows = torch.tensor(table.row_indices, dtype=torch.long, device=device)
        columns = torch.tensor(
            table.column_indices,
            dtype=torch.long,
            device=device,
        )
        values = torch.tensor(table.values, dtype=dtype, device=device)
        matrix.index_put_((rows, columns), values, accumulate=True)
    return matrix


def _symmetric_power_plan_key(plan):
    """Return the canonical numeric identity of one power contraction."""

    payload = {
        "descriptor_count": int(plan.descriptor_count),
        "channel_count": int(plan.channel_count),
        "carrier": str(plan.carrier),
        "factor_basis": str(plan.factor_basis),
        "normalization_convention": str(plan.normalization_convention),
        "entries": [
            {
                "descriptor_index": int(entry.descriptor_index),
                "channel_indices": [
                    int(value) for value in entry.channel_indices
                ],
                "power": int(entry.power),
                "input_L": int(entry.input_L),
                "output_L": int(entry.output_L),
                "multiplicity_index": int(entry.multiplicity_index),
                "component_index": int(entry.component_index),
                "component_terms": [
                    {
                        "exponents": [
                            int(value) for value in term["exponents"]
                        ],
                        "coefficient": [
                            float(complex(term["coefficient"]).real),
                            float(complex(term["coefficient"]).imag),
                        ],
                    }
                    for term in entry.component_terms
                ],
            }
            for entry in plan.entries
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class YE3THierarchicalRepeatedBlockModule(torch.nn.Module):
    """Evaluate one factored Schur/LR repeated-angular execution instruction."""

    def __init__(
        self,
        execution_plan,
        instruction_id=None,
        backend="auto",
        dtype=None,
        device=None,
    ):
        super().__init__()
        if not isinstance(execution_plan, YE3TExecutionPlan):
            execution_plan = YE3TExecutionPlan.from_dict(execution_plan)
        instructions = {
            instruction.instruction_id: instruction
            for instruction in execution_plan.instructions
        }
        if instruction_id is None:
            if len(execution_plan.forward_schedule) != 1:
                raise ValueError(
                    "instruction_id is required for a multi-instruction plan"
                )
            instruction_id = execution_plan.forward_schedule[0]
        instruction_id = str(instruction_id)
        if instruction_id not in instructions:
            raise ValueError("instruction_id is not present in the execution plan")
        instruction = instructions[instruction_id]
        metadata = dict(instruction.metadata)
        if instruction.opcode != "block_symmetric_power":
            raise ValueError(
                "hierarchical repeated-block runtime requires block_symmetric_power"
            )
        hierarchical_schema = metadata.get("schema")
        if hierarchical_schema not in {
            "ye3t_hierarchical_repeated_angular_blocks_v1",
            "ye3t_hierarchical_repeated_angular_blocks_v2",
        }:
            raise ValueError("unsupported hierarchical repeated-block schema")
        if bool(metadata.get("runtime_path_discovery", True)):
            raise ValueError("hierarchical runtime forbids runtime path discovery")
        if bool(metadata.get("raw_angular_tree_forest_materialized", True)):
            raise ValueError("hierarchical runtime forbids raw angular tree forests")
        backend = str(backend)
        if backend not in {"auto", "native", "reference"}:
            raise ValueError("backend must be auto, native, or reference")
        if dtype is None:
            dtype = torch.complex128
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError(
                "hierarchical complex-magnetic plans require a complex dtype"
            )
        device = torch.device("cpu" if device is None else device)

        blocks = tuple(dict(block) for block in metadata["blocks"])
        block_power_plans = {}
        for record in metadata["block_power_plans"]:
            record = dict(record)
            key = (int(record["block_index"]), int(record["output_L"]))
            if key in block_power_plans:
                raise ValueError("duplicate hierarchical block power plan")
            plan = SymmetricPowerProductPlan.from_dict(record["plan"])
            expected_width = 2 * int(blocks[key[0]]["input_L"]) + 1
            if int(plan.channel_count) != expected_width:
                raise ValueError("hierarchical block power input width is inconsistent")
            block_power_plans[key] = plan
        if not block_power_plans:
            raise ValueError("hierarchical instruction has no block power plans")
        object.__setattr__(self, "_block_power_plans", block_power_plans)
        object.__setattr__(
            self,
            "_block_power_table_reports",
            tuple(
                {
                    "block_index": int(key[0]),
                    "output_L": int(key[1]),
                    **symmetric_power_product_plan_table_report(
                        plan,
                        device=device,
                        dtype=dtype,
                    ),
                }
                for key, plan in sorted(block_power_plans.items())
            ),
        )

        tables = {
            table.table_id: table for table in execution_plan.synthesis_tables
        }
        lr_table_id = str(metadata["lr_synthesis_table_id"])
        if lr_table_id not in tables:
            raise ValueError("hierarchical LR synthesis table is missing")
        self.register_buffer(
            "lr_synthesis",
            _dense_synthesis_table(tables[lr_table_id], dtype, device),
            persistent=False,
        )
        routes = tuple(dict(route) for route in metadata["routes"])
        if tuple(int(route["route_index"]) for route in routes) != tuple(
            range(len(routes))
        ):
            raise ValueError("hierarchical routes must use canonical contiguous indices")
        cg_buffer_names = {}
        cg_sparse_entries = {}
        for table_id in sorted(
            {
                str(route["angular_synthesis_table_id"])
                for route in routes
                if route.get("angular_synthesis_table_id") is not None
            }
        ):
            if table_id not in tables:
                raise ValueError("hierarchical angular synthesis table is missing")
            table = tables[table_id]
            cg_sparse_entries[table_id] = {
                "input_dimension": int(table.input_dimension),
                "output_dimension": int(table.output_dimension),
                "rows": tuple(int(value) for value in table.row_indices),
                "columns": tuple(
                    int(value) for value in table.column_indices
                ),
                "values": tuple(complex(value) for value in table.values),
            }
            if hierarchical_schema == (
                "ye3t_hierarchical_repeated_angular_blocks_v1"
            ):
                buffer_name = "cg_synthesis_" + str(len(cg_buffer_names))
                self.register_buffer(
                    buffer_name,
                    _dense_synthesis_table(table, dtype, device),
                    persistent=False,
                )
                cg_buffer_names[table_id] = buffer_name
        object.__setattr__(self, "_cg_buffer_names", cg_buffer_names)
        object.__setattr__(self, "_cg_sparse_entries", cg_sparse_entries)
        object.__setattr__(self, "_routes", routes)
        object.__setattr__(self, "_blocks", blocks)

        layouts = tuple(
            layout
            for layout in execution_plan.carrier_layouts
            if layout.key == instruction.output_carrier
        )
        if len(layouts) != 1:
            raise ValueError("hierarchical instruction must have one output layout")
        layout = layouts[0]
        self.instruction_id = instruction_id
        self.backend = backend
        self.plan_hash = str(execution_plan.plan_hash)
        self.coefficient_hash = str(execution_plan.coefficient_hash)
        self.channel_count = int(layout.channel_count)
        self.tableau_count = int(layout.tableau_count)
        self.magnetic_count = int(layout.magnetic_count)
        self.output_dimension = int(layout.width)
        self.target_L = int(metadata["target_L"])
        self.target_parity = metadata.get("target_parity")
        self.lr_multiplicity = int(metadata["lr_multiplicity"])
        self.block_count = int(len(blocks))
        self.route_count = int(len(routes))
        self.hierarchical_schema = str(hierarchical_schema)
        self.last_backend = None
        self.last_linear_readout_backend = None
        if (
            self.hierarchical_schema
            == "ye3t_hierarchical_repeated_angular_blocks_v1"
            and self.block_count not in {1, 2}
        ):
            raise ValueError(
                "hierarchical repeated-block runtime currently requires one "
                "or two compiler-factored blocks"
            )
        if self.channel_count != self.route_count * self.lr_multiplicity:
            raise ValueError("hierarchical route/LR count does not match output layout")
        if tuple(self.lr_synthesis.shape) != (
            self.lr_multiplicity,
            self.tableau_count,
        ):
            raise ValueError("hierarchical LR synthesis dimensions are inconsistent")

        if (
            self.hierarchical_schema
            == "ye3t_hierarchical_repeated_angular_blocks_v2"
        ):
            generic_routes = []
            for route_index, route in enumerate(self._routes):
                table_id = str(route["angular_synthesis_table_id"])
                sparse = self._cg_sparse_entries[table_id]
                output_Ls = tuple(
                    int(value) for value in route["block_output_Ls"]
                )
                multiplicities = tuple(
                    int(value)
                    for value in route["block_multiplicity_indices"]
                )
                widths = tuple(2 * value + 1 for value in output_Ls)
                rows = tuple(int(value) for value in sparse["rows"])
                component_columns = []
                for block_index, width in enumerate(widths):
                    stride = math.prod(widths[block_index + 1 :])
                    local = tuple(
                        (row // int(stride)) % int(width) for row in rows
                    )
                    name = (
                        "generic_route_"
                        + str(route_index)
                        + "_block_"
                        + str(block_index)
                    )
                    self.register_buffer(
                        name,
                        torch.tensor(
                            tuple(
                                int(multiplicities[block_index]) * int(width)
                                + int(component)
                                for component in local
                            ),
                            dtype=torch.long,
                            device=device,
                        ),
                        persistent=False,
                    )
                    component_columns.append(name)
                output_name = "generic_route_" + str(route_index) + "_output"
                coefficient_name = (
                    "generic_route_" + str(route_index) + "_coefficient"
                )
                self.register_buffer(
                    output_name,
                    torch.tensor(
                        sparse["columns"],
                        dtype=torch.long,
                        device=device,
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    coefficient_name,
                    torch.tensor(
                        tuple(value.conjugate() for value in sparse["values"]),
                        dtype=dtype,
                        device=device,
                    ),
                    persistent=False,
                )
                generic_routes.append(
                    {
                        "route_index": int(route_index),
                        "output_Ls": output_Ls,
                        "component_columns": tuple(component_columns),
                        "output_indices": output_name,
                        "coefficients": coefficient_name,
                    }
                )
            object.__setattr__(self, "_generic_routes", tuple(generic_routes))
            object.__setattr__(self, "_route_groups", tuple())
        elif self.block_count == 1:
            self.register_buffer(
                "single_block_route_indices",
                torch.tensor(
                    tuple(
                        int(route["block_multiplicity_indices"][0])
                        for route in self._routes
                    ),
                    dtype=torch.long,
                    device=device,
                ),
                persistent=False,
            )
            object.__setattr__(self, "_route_groups", tuple())
        else:
            routes_by_table = {}
            for route in self._routes:
                table_id = str(route["angular_synthesis_table_id"])
                routes_by_table.setdefault(table_id, []).append(route)
            route_groups = []
            for group_index, table_id in enumerate(sorted(routes_by_table)):
                table_routes = tuple(routes_by_table[table_id])
                left_name = "route_left_indices_" + str(group_index)
                right_name = "route_right_indices_" + str(group_index)
                output_name = "route_output_indices_" + str(group_index)
                self.register_buffer(
                    left_name,
                    torch.tensor(
                        tuple(
                            int(route["block_multiplicity_indices"][0])
                            for route in table_routes
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    right_name,
                    torch.tensor(
                        tuple(
                            int(route["block_multiplicity_indices"][1])
                            for route in table_routes
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    output_name,
                    torch.tensor(
                        tuple(int(route["route_index"]) for route in table_routes),
                        dtype=torch.long,
                        device=device,
                    ),
                    persistent=False,
                )
                route_groups.append(
                    {
                        "table_id": table_id,
                        "left_L": int(table_routes[0]["block_output_Ls"][0]),
                        "right_L": int(table_routes[0]["block_output_Ls"][1]),
                        "route_count": int(len(table_routes)),
                        "left_indices": left_name,
                        "right_indices": right_name,
                        "output_indices": output_name,
                    }
                )
            object.__setattr__(self, "_route_groups", tuple(route_groups))
            object.__setattr__(self, "_generic_routes", tuple())

    def block_power_requests(self):
        return tuple(
            {
                "block_index": int(block_index),
                "output_L": int(output_L),
                "plan_key": _symmetric_power_plan_key(plan),
                "plan": plan,
            }
            for (block_index, output_L), plan in sorted(
                self._block_power_plans.items()
            )
        )

    def _evaluate_block_plans(self, block_inputs):
        evaluated = {}
        for key, plan in self._block_power_plans.items():
            block_index, output_L = key
            value = symmetric_power_product_plan_contraction(
                block_inputs[block_index],
                plan,
                backend=self.backend,
            )
            magnetic_count = 2 * int(output_L) + 1
            if int(plan.descriptor_count) % magnetic_count:
                raise ValueError("block power descriptor layout is not multiplet-complete")
            multiplicity = int(plan.descriptor_count) // magnetic_count
            evaluated[key] = value.reshape(
                tuple(value.shape[:-1]) + (multiplicity, magnetic_count)
            )
        return evaluated

    def _evaluate_routes(self, evaluated, leading_shape):
        if (
            self.hierarchical_schema
            == "ye3t_hierarchical_repeated_angular_blocks_v2"
        ):
            route_values = []
            for route in self._generic_routes:
                term_values = getattr(self, route["coefficients"])
                term_values = term_values.reshape(
                    (1,) * len(leading_shape) + (-1,)
                )
                for block_index, output_L in enumerate(route["output_Ls"]):
                    block = evaluated[(int(block_index), int(output_L))]
                    block = block.reshape(tuple(leading_shape) + (-1,))
                    indices = getattr(
                        self,
                        route["component_columns"][block_index],
                    )
                    term_values = term_values * block.index_select(-1, indices)
                output = torch.zeros(
                    tuple(leading_shape) + (self.magnetic_count,),
                    dtype=self.lr_synthesis.dtype,
                    device=self.lr_synthesis.device,
                )
                output.index_add_(
                    -1,
                    getattr(self, route["output_indices"]),
                    term_values,
                )
                route_values.append(output)
            return torch.stack(route_values, dim=-2)
        if self.block_count == 1:
            value = evaluated[(0, self.target_L)]
            return value.index_select(-2, self.single_block_route_indices)

        route_output = torch.empty(
            tuple(leading_shape) + (self.route_count, self.magnetic_count),
            dtype=self.lr_synthesis.dtype,
            device=self.lr_synthesis.device,
        )
        route_axis = len(leading_shape)
        for group in self._route_groups:
            table_id = str(group["table_id"])
            left_L = int(group["left_L"])
            right_L = int(group["right_L"])
            left = evaluated[(0, left_L)]
            right = evaluated[(1, right_L)]
            left_indices = getattr(self, group["left_indices"])
            right_indices = getattr(self, group["right_indices"])
            left_selected = left.index_select(-2, left_indices)
            right_selected = right.index_select(-2, right_indices)
            product = (
                left_selected.unsqueeze(-1) * right_selected.unsqueeze(-2)
            ).reshape(
                tuple(leading_shape) + (int(group["route_count"]), -1)
            )
            cg = getattr(self, self._cg_buffer_names[table_id])
            coupled = torch.matmul(product, cg.conj())
            route_indices = getattr(self, group["output_indices"])
            route_output.index_copy_(route_axis, route_indices, coupled)
        return route_output

    def _forward_from_evaluated(self, evaluated, leading_shape):
        route_values = self._evaluate_routes(evaluated, leading_shape)
        lr_analysis = self.lr_synthesis.conj()
        output = (
            route_values.unsqueeze(-2).unsqueeze(-2)
            * lr_analysis.reshape(
                (1,) * len(leading_shape)
                + (1, self.lr_multiplicity, self.tableau_count, 1)
            )
        )
        return output.reshape(
            tuple(leading_shape)
            + (self.channel_count, self.tableau_count, self.magnetic_count)
        )

    def forward(self, *block_inputs):
        if len(block_inputs) == 1 and isinstance(block_inputs[0], (tuple, list)):
            block_inputs = tuple(block_inputs[0])
        if len(block_inputs) != self.block_count:
            raise ValueError("hierarchical input count does not match block count")
        converted = []
        leading_shape = None
        for block, value in zip(self._blocks, block_inputs):
            value = torch.as_tensor(
                value,
                dtype=self.lr_synthesis.dtype,
                device=self.lr_synthesis.device,
            )
            expected_width = 2 * int(block["input_L"]) + 1
            if int(value.shape[-1]) != expected_width:
                raise ValueError("hierarchical block input multiplet width is incorrect")
            current_leading = tuple(value.shape[:-1])
            if leading_shape is None:
                leading_shape = current_leading
            elif current_leading != leading_shape:
                raise ValueError("hierarchical block inputs require matching leading shapes")
            converted.append(value)
        evaluated = self._evaluate_block_plans(tuple(converted))
        output = self._forward_from_evaluated(evaluated, leading_shape)
        if (
            self.hierarchical_schema
            == "ye3t_hierarchical_repeated_angular_blocks_v2"
        ):
            self.last_backend = (
                "shared_monomial_then_sparse_factorized_outer_then_lr_analysis"
            )
        else:
            self.last_backend = (
                "shared_monomial_then_grouped_binary_cg_then_lr_analysis"
            )
        return output

    def forward_flat(self, *block_inputs):
        output = self.forward(*block_inputs)
        return output.reshape(tuple(output.shape[:-3]) + (self.output_dimension,))

    def linear_readout(self, source, weight, bias):
        block_inputs = tuple(source) if isinstance(source, (tuple, list)) else (source,)
        features = self.forward_flat(*block_inputs)
        weight = torch.as_tensor(
            weight,
            dtype=features.dtype,
            device=features.device,
        )
        bias = torch.as_tensor(
            bias,
            dtype=features.dtype,
            device=features.device,
        ).reshape(())
        if weight.ndim != 1 or int(weight.numel()) != self.output_dimension:
            raise ValueError("hierarchical readout weight has the wrong width")
        self.last_linear_readout_backend = "materialized_hierarchical_readout"
        return features @ weight + bias

    def runtime_report(self):
        return {
            "runtime": "YE3THierarchicalRepeatedBlockModule",
            "instruction_id": str(self.instruction_id),
            "plan_hash": str(self.plan_hash),
            "coefficient_hash": str(self.coefficient_hash),
            "block_count": int(self.block_count),
            "route_count": int(self.route_count),
            "channel_count": int(self.channel_count),
            "tableau_count": int(self.tableau_count),
            "magnetic_count": int(self.magnetic_count),
            "target_L": int(self.target_L),
            "target_parity": self.target_parity,
            "backend": str(self.backend),
            "last_backend": self.last_backend,
            "last_linear_readout_backend": self.last_linear_readout_backend,
            "runtime_path_discovery": False,
            "raw_angular_tree_forest_materialized": False,
            "composition": (
                "shared_block_power_then_"
                + (
                    "sparse_factorized_outer_C_dagger_then_LR_C_dagger"
                    if self.hierarchical_schema
                    == "ye3t_hierarchical_repeated_angular_blocks_v2"
                    else "grouped_binary_C_dagger_then_LR_C_dagger"
                )
            ),
            "block_power_table_reports": tuple(
                dict(record)
                for record in self._block_power_table_reports
            ),
            "supports_autograd_vjp_hvp": True,
        }


class YE3THierarchicalRepeatedBlockModuleGroup(torch.nn.Module):
    """Reuse block powers and optionally segment exact parent destinations."""

    def __init__(
        self,
        modules,
        block_binding_indices_by_module,
        parent_execution_policy="reference",
        block_power_execution_policy="individual",
    ):
        super().__init__()
        modules = tuple(modules)
        if not modules:
            raise ValueError("hierarchical module group requires at least one module")
        if any(
            not isinstance(module, YE3THierarchicalRepeatedBlockModule)
            for module in modules
        ):
            raise ValueError(
                "hierarchical module group accepts only compiler-owned "
                "hierarchical modules"
            )
        binding_rows = tuple(
            tuple(int(value) for value in row)
            for row in block_binding_indices_by_module
        )
        if len(binding_rows) != len(modules):
            raise ValueError(
                "hierarchical block binding rows must match the module count"
            )
        for module, row in zip(modules, binding_rows):
            if len(row) != int(module.block_count):
                raise ValueError(
                    "hierarchical block binding count must match compiler blocks"
                )
            if any(value < 0 for value in row):
                raise ValueError("hierarchical block binding indices must be nonnegative")
        binding_indices = tuple(
            sorted({value for row in binding_rows for value in row})
        )
        if binding_indices != tuple(range(len(binding_indices))):
            raise ValueError(
                "hierarchical block binding indices must be canonical and contiguous"
            )
        first = modules[0]
        if any(
            module.lr_synthesis.dtype != first.lr_synthesis.dtype
            or module.lr_synthesis.device != first.lr_synthesis.device
            or module.backend != first.backend
            for module in modules[1:]
        ):
            raise ValueError(
                "hierarchical modules must share backend, dtype, and device"
            )
        object.__setattr__(self, "_source_modules", modules)
        object.__setattr__(self, "_binding_rows", binding_rows)

        unique_power_records = []
        unique_power_lookup = {}
        module_power_indices = []
        power_request_count = 0
        for module_index, (module, binding_row) in enumerate(
            zip(modules, binding_rows)
        ):
            current = {}
            for request in module.block_power_requests():
                block_index = int(request["block_index"])
                output_L = int(request["output_L"])
                unique_key = (
                    int(binding_row[block_index]),
                    str(request["plan_key"]),
                )
                if unique_key not in unique_power_lookup:
                    unique_power_lookup[unique_key] = len(unique_power_records)
                    unique_power_records.append(
                        {
                            "binding_index": int(binding_row[block_index]),
                            "output_L": output_L,
                            "plan": request["plan"],
                            "plan_key": str(request["plan_key"]),
                        }
                    )
                current[(block_index, output_L)] = int(
                    unique_power_lookup[unique_key]
                )
                power_request_count += 1
            module_power_indices.append(current)
        object.__setattr__(self, "_unique_power_records", tuple(unique_power_records))
        object.__setattr__(self, "_module_power_indices", tuple(module_power_indices))

        parent_execution_policy = str(parent_execution_policy)
        if parent_execution_policy not in {
            "auto",
            "reference",
            "destination_segmented",
        }:
            raise ValueError(
                "parent_execution_policy must be auto, reference, or "
                "destination_segmented"
            )
        destination_native_capable = False
        native_error = None
        if parent_execution_policy != "reference":
            try:
                _extension_for_native_dispatch()
            except RuntimeError as error:
                native_error = error
            else:
                destination_native_capable = hasattr(
                    torch.ops.ye3t_runtime,
                    "factorized_angular_segmented",
                )
        if (
            parent_execution_policy == "destination_segmented"
            and not destination_native_capable
        ):
            message = (
                "destination-segmented hierarchical execution requires the "
                "YE3T native factorized-angular runtime"
            )
            if native_error is not None:
                raise RuntimeError(message) from native_error
            raise RuntimeError(message)
        self.parent_execution_policy_requested = parent_execution_policy
        self.destination_native_capable = bool(destination_native_capable)
        self.destination_native_enabled = bool(
            destination_native_capable
            and parent_execution_policy
            in {"auto", "destination_segmented"}
        )
        block_power_execution_policy = str(block_power_execution_policy)
        if block_power_execution_policy not in {
            "individual",
            "shared_monomial",
        }:
            raise ValueError(
                "block_power_execution_policy must be individual or "
                "shared_monomial"
            )
        self.block_power_execution_policy = block_power_execution_policy

        output_slices = []
        output_base = 0
        for module_index, module in enumerate(modules):
            output_slices.append(
                {
                    "module_index": int(module_index),
                    "start": int(output_base),
                    "stop": int(output_base + module.output_dimension),
                    "instruction_id": str(module.instruction_id),
                    "plan_hash": str(module.plan_hash),
                }
            )
            output_base += int(module.output_dimension)
        self.output_slices = tuple(output_slices)
        self.output_dimension = int(output_base)
        self.unique_binding_count = int(len(binding_indices))
        self.power_request_count = int(power_request_count)
        self.power_evaluation_count = int(len(unique_power_records))
        self.shared_power_group_count = 0
        self.last_backend = None
        if self.block_power_execution_policy == "shared_monomial":
            self._prepare_shared_power_groups()
        if self.destination_native_enabled:
            self._prepare_destination_segmented_schedule()

    def _prepare_shared_power_groups(self):
        first = self._source_modules[0]
        device = first.lr_synthesis.device
        dtype = first.lr_synthesis.dtype
        indices_by_binding = {}
        for power_index, record in enumerate(self._unique_power_records):
            indices_by_binding.setdefault(
                int(record["binding_index"]),
                [],
            ).append(int(power_index))
        groups = []
        for group_index, binding_index in enumerate(
            sorted(indices_by_binding)
        ):
            power_indices = tuple(indices_by_binding[binding_index])
            plans = tuple(
                self._unique_power_records[index]["plan"]
                for index in power_indices
            )
            channel_counts = {
                int(plan.channel_count) for plan in plans
            }
            if len(channel_counts) != 1:
                raise ValueError(
                    "shared hierarchical power bindings require one input "
                    "channel dimension"
                )
            count_rows = []
            output_offsets = [0]
            coefficient_outputs = []
            coefficient_values = []
            output_slices = []
            term_base = 0
            output_base = 0
            for power_index, plan in zip(power_indices, plans):
                grouped, _shared, output_dtype = (
                    _symmetric_power_product_plan_count_table(
                        plan,
                        device=device,
                        dtype=dtype,
                        imag_tol=1.0e-12,
                    )
                )
                if output_dtype != dtype:
                    raise ValueError(
                        "hierarchical shared powers require one complex "
                        "coefficient dtype"
                    )
                local_counts, local_offsets, local_outputs, local_values = (
                    grouped
                )
                count_rows.extend(
                    tuple(int(value) for value in row)
                    for row in local_counts.detach().cpu().tolist()
                )
                output_offsets.extend(
                    int(term_base + value)
                    for value in local_offsets.detach().cpu().tolist()[1:]
                )
                coefficient_outputs.extend(
                    int(output_base + value)
                    for value in local_outputs.detach().cpu().tolist()
                )
                coefficient_values.append(local_values.to(dtype=dtype))
                output_slices.append(
                    {
                        "power_index": int(power_index),
                        "start": int(output_base),
                        "stop": int(
                            output_base + int(plan.descriptor_count)
                        ),
                    }
                )
                term_base += int(local_counts.shape[0])
                output_base += int(plan.descriptor_count)
            values = torch.cat(tuple(coefficient_values))
            shared = _shared_symmetric_power_table(
                count_rows,
                output_offsets,
                coefficient_outputs,
                values,
                next(iter(channel_counts)),
                device,
            )
            names = []
            for suffix, value in zip(
                (
                    "monomial_counts",
                    "output_offsets",
                    "coefficient_terms",
                    "coefficient_outputs",
                    "coefficient_values",
                ),
                shared,
            ):
                name = (
                    "shared_power_"
                    + str(group_index)
                    + "_"
                    + str(suffix)
                )
                self.register_buffer(name, value, persistent=False)
                names.append(name)
            groups.append(
                {
                    "binding_index": int(binding_index),
                    "power_indices": power_indices,
                    "buffer_names": tuple(names),
                    "output_slices": tuple(output_slices),
                    "input_dimension": int(next(iter(channel_counts))),
                    "output_dimension": int(output_base),
                    "coefficient_count": int(values.numel()),
                    "unique_monomial_count": int(shared[0].shape[0]),
                }
            )
        object.__setattr__(self, "_shared_power_groups", tuple(groups))
        self.shared_power_group_count = int(len(groups))

    def _prepare_destination_segmented_schedule(self):
        modules = self._source_modules
        first = modules[0]
        device = first.lr_synthesis.device
        dtype = first.lr_synthesis.dtype
        power_offsets = []
        power_dimension = 0
        for record in self._unique_power_records:
            power_offsets.append(int(power_dimension))
            power_dimension += int(record["plan"].descriptor_count)
        object.__setattr__(self, "_power_output_offsets", tuple(power_offsets))
        self.power_output_dimension = int(power_dimension)

        segment_source_offsets = [0]
        segment_input_offsets = []
        segment_input_dimensions = []
        segment_workspace_offsets = []
        segment_workspace_dimensions = []
        segment_node_offsets = [0]
        segment_root_offsets = [0]
        segment_projection_offsets = []
        segment_projection_dimensions = []
        segment_output_offsets = [0]
        node_offsets = []
        node_dimensions = []
        node_leaf_offsets = []
        node_left = []
        node_right = []
        node_coefficient_offsets = []
        coefficient_rows = []
        coefficient_columns = []
        coefficient_values = []
        root_nodes = []
        root_projection_starts = []
        root_projection_dimensions = []
        root_output_offsets = []
        projection_values = []
        gather_indices = []
        segment_reports = []
        expanded_output_slices = []
        input_base = 0
        workspace_base = 0
        coefficient_base = 0
        projection_base = 0
        output_base = 0
        sentinel = int(power_dimension)

        for module_index, (module, power_indices) in enumerate(
            zip(modules, self._module_power_indices)
        ):
            source_dimension = int(module.route_count)
            logical_dimension = int(
                module.channel_count * module.tableau_count
            )
            if module.block_count == 1:
                left_input_width = int(module.magnetic_count)
                right_input_width = 0
                route_groups = (
                    {
                        "route_indices": tuple(range(source_dimension)),
                        "left_L": int(module.target_L),
                        "right_L": None,
                        "table_id": None,
                    },
                )
            else:
                left_input_width = max(
                    2 * int(route["block_output_Ls"][0]) + 1
                    for route in module._routes
                )
                right_input_width = max(
                    2 * int(route["block_output_Ls"][1]) + 1
                    for route in module._routes
                )
                route_groups = tuple(
                    {
                        "route_indices": tuple(
                            int(value)
                            for value in getattr(
                                module,
                                group["output_indices"],
                            ).detach().cpu().tolist()
                        ),
                        "left_L": int(group["left_L"]),
                        "right_L": int(group["right_L"]),
                        "table_id": str(group["table_id"]),
                    }
                    for group in module._route_groups
                )
            input_dimension = int(left_input_width + right_input_width)
            local_gather = [sentinel] * (
                source_dimension * input_dimension
            )
            for route in module._routes:
                route_index = int(route["route_index"])
                block_output_Ls = tuple(
                    int(value) for value in route["block_output_Ls"]
                )
                multiplicity_indices = tuple(
                    int(value)
                    for value in route["block_multiplicity_indices"]
                )
                for block_index, (output_L, multiplicity_index) in enumerate(
                    zip(block_output_Ls, multiplicity_indices)
                ):
                    power_index = int(
                        power_indices[(block_index, output_L)]
                    )
                    magnetic_dimension = 2 * output_L + 1
                    source_start = int(
                        power_offsets[power_index]
                        + multiplicity_index * magnetic_dimension
                    )
                    destination_start = int(
                        route_index * input_dimension
                        + (0 if block_index == 0 else left_input_width)
                    )
                    for magnetic in range(magnetic_dimension):
                        local_gather[destination_start + magnetic] = int(
                            source_start + magnetic
                        )
            gather_indices.extend(local_gather)

            local_node_offsets = []
            local_node_dimensions = []
            local_node_leaf_offsets = []
            local_node_left = []
            local_node_right = []
            local_coefficient_offsets = [0]
            local_coefficient_rows = []
            local_coefficient_columns = []
            local_coefficient_values = []
            local_root_nodes = []
            workspace_cursor = 0
            leaf_nodes = {}

            def add_leaf(offset, dimension):
                nonlocal workspace_cursor
                key = (int(offset), int(dimension))
                if key in leaf_nodes:
                    return int(leaf_nodes[key])
                node = len(local_node_offsets)
                leaf_nodes[key] = int(node)
                local_node_offsets.append(int(workspace_cursor))
                local_node_dimensions.append(int(dimension))
                local_node_leaf_offsets.append(int(offset))
                local_node_left.append(-1)
                local_node_right.append(-1)
                local_coefficient_offsets.append(
                    len(local_coefficient_values)
                )
                workspace_cursor += int(dimension)
                return int(node)

            for group in route_groups:
                left_dimension = 2 * int(group["left_L"]) + 1
                left_node = add_leaf(0, left_dimension)
                if group["right_L"] is None:
                    root_node = int(left_node)
                else:
                    right_dimension = 2 * int(group["right_L"]) + 1
                    right_node = add_leaf(
                        left_input_width,
                        right_dimension,
                    )
                    sparse = module._cg_sparse_entries[
                        str(group["table_id"])
                    ]
                    if int(sparse["input_dimension"]) != int(
                        left_dimension * right_dimension
                    ) or int(sparse["output_dimension"]) != int(
                        module.magnetic_count
                    ):
                        raise ValueError(
                            "hierarchical angular table dimensions do not "
                            "match the compiled route"
                        )
                    root_node = len(local_node_offsets)
                    local_node_offsets.append(int(workspace_cursor))
                    local_node_dimensions.append(int(module.magnetic_count))
                    local_node_leaf_offsets.append(-1)
                    local_node_left.append(int(left_node))
                    local_node_right.append(int(right_node))
                    local_coefficient_rows.extend(sparse["rows"])
                    local_coefficient_columns.extend(sparse["columns"])
                    local_coefficient_values.extend(sparse["values"])
                    local_coefficient_offsets.append(
                        len(local_coefficient_values)
                    )
                    workspace_cursor += int(module.magnetic_count)
                local_root_nodes.append(int(root_node))
            local_workspace_dimension = int(workspace_cursor)

            projection_stride = int(
                len(route_groups) * logical_dimension
            )
            local_projection = torch.zeros(
                (source_dimension, projection_stride),
                dtype=dtype,
                device=device,
            )
            lr_analysis = module.lr_synthesis.conj()
            for root_index, group in enumerate(route_groups):
                root_projection_starts.append(
                    int(root_index * logical_dimension)
                )
                root_projection_dimensions.append(int(logical_dimension))
                root_output_offsets.append(
                    int(root_index * module.output_dimension)
                )
                for route_index in group["route_indices"]:
                    for multiplicity in range(module.lr_multiplicity):
                        for tableau in range(module.tableau_count):
                            logical_index = int(
                                (
                                    route_index * module.lr_multiplicity
                                    + multiplicity
                                )
                                * module.tableau_count
                                + tableau
                            )
                            local_projection[
                                route_index,
                                root_index * logical_dimension
                                + logical_index,
                            ] = lr_analysis[multiplicity, tableau]

            shifted_offsets = torch.tensor(
                tuple(
                    int(value + coefficient_base)
                    for value in local_coefficient_offsets
                ),
                dtype=torch.int64,
                device=device,
            )
            if node_coefficient_offsets:
                shifted_offsets = shifted_offsets[1:]
            node_coefficient_offsets.append(shifted_offsets)
            coefficient_base += len(local_coefficient_values)
            node_offsets.extend(local_node_offsets)
            node_dimensions.extend(local_node_dimensions)
            node_leaf_offsets.extend(local_node_leaf_offsets)
            node_left.extend(local_node_left)
            node_right.extend(local_node_right)
            coefficient_rows.extend(local_coefficient_rows)
            coefficient_columns.extend(local_coefficient_columns)
            coefficient_values.extend(local_coefficient_values)
            root_nodes.extend(local_root_nodes)
            projection_values.append(local_projection.reshape(-1))
            segment_source_offsets.append(
                int(segment_source_offsets[-1] + source_dimension)
            )
            segment_input_offsets.append(int(input_base))
            segment_input_dimensions.append(int(input_dimension))
            segment_workspace_offsets.append(int(workspace_base))
            segment_workspace_dimensions.append(
                int(local_workspace_dimension)
            )
            segment_node_offsets.append(
                int(segment_node_offsets[-1] + len(local_node_offsets))
            )
            segment_root_offsets.append(
                int(segment_root_offsets[-1] + len(local_root_nodes))
            )
            segment_projection_offsets.append(int(projection_base))
            segment_projection_dimensions.append(int(projection_stride))
            expanded_output_dimension = int(
                len(local_root_nodes) * module.output_dimension
            )
            segment_output_offsets.append(
                int(output_base + expanded_output_dimension)
            )
            expanded_output_slices.append(
                {
                    "module_index": int(module_index),
                    "start": int(output_base),
                    "stop": int(output_base + expanded_output_dimension),
                    "root_count": int(len(local_root_nodes)),
                    "output_dimension": int(module.output_dimension),
                }
            )
            segment_reports.append(
                {
                    "module_index": int(module_index),
                    "rank": int(sum(int(block["power"]) for block in module._blocks)),
                    "source_route_count": int(source_dimension),
                    "input_dimension": int(input_dimension),
                    "root_count": int(len(local_root_nodes)),
                    "node_count": int(len(local_node_offsets)),
                    "coefficient_count": int(len(local_coefficient_values)),
                    "projection_nonzero_count": int(
                        torch.count_nonzero(local_projection).item()
                    ),
                    "output_dimension": int(module.output_dimension),
                    "expanded_output_dimension": int(
                        expanded_output_dimension
                    ),
                }
            )
            input_base += int(source_dimension * input_dimension)
            workspace_base += int(
                source_dimension * local_workspace_dimension
            )
            projection_base += int(local_projection.numel())
            output_base += int(expanded_output_dimension)

        def register_integer(name, values):
            self.register_buffer(
                name,
                torch.as_tensor(
                    values,
                    dtype=torch.int64,
                    device=device,
                ).contiguous(),
                persistent=False,
            )

        register_integer(
            "destination_power_gather_indices",
            gather_indices,
        )
        register_integer(
            "destination_segment_source_offsets",
            segment_source_offsets,
        )
        register_integer(
            "destination_segment_input_offsets",
            segment_input_offsets,
        )
        register_integer(
            "destination_segment_input_dimensions",
            segment_input_dimensions,
        )
        register_integer(
            "destination_segment_workspace_offsets",
            segment_workspace_offsets,
        )
        register_integer(
            "destination_segment_workspace_dimensions",
            segment_workspace_dimensions,
        )
        register_integer(
            "destination_segment_node_offsets",
            segment_node_offsets,
        )
        register_integer(
            "destination_segment_root_offsets",
            segment_root_offsets,
        )
        register_integer(
            "destination_segment_projection_offsets",
            segment_projection_offsets,
        )
        register_integer(
            "destination_segment_projection_dimensions",
            segment_projection_dimensions,
        )
        register_integer(
            "destination_segment_output_offsets",
            segment_output_offsets,
        )
        register_integer("destination_node_offsets", node_offsets)
        register_integer("destination_node_dimensions", node_dimensions)
        register_integer(
            "destination_node_leaf_offsets",
            node_leaf_offsets,
        )
        register_integer("destination_node_left", node_left)
        register_integer("destination_node_right", node_right)
        self.register_buffer(
            "destination_node_coefficient_offsets",
            torch.cat(tuple(node_coefficient_offsets)).contiguous(),
            persistent=False,
        )
        register_integer("destination_coefficient_rows", coefficient_rows)
        register_integer(
            "destination_coefficient_columns",
            coefficient_columns,
        )
        self.register_buffer(
            "destination_coefficient_values",
            torch.as_tensor(
                coefficient_values,
                dtype=dtype,
                device=device,
            ).contiguous(),
            persistent=False,
        )
        register_integer("destination_root_nodes", root_nodes)
        register_integer(
            "destination_root_projection_starts",
            root_projection_starts,
        )
        register_integer(
            "destination_root_projection_dimensions",
            root_projection_dimensions,
        )
        register_integer(
            "destination_root_output_offsets",
            root_output_offsets,
        )
        self.register_buffer(
            "destination_projection_values",
            torch.cat(tuple(projection_values)).contiguous(),
            persistent=False,
        )
        self.destination_total_source_count = int(
            segment_source_offsets[-1]
        )
        self.destination_input_dimension = int(input_base)
        self.destination_workspace_dimension = int(workspace_base)
        self.destination_output_dimension = int(output_base)
        object.__setattr__(
            self,
            "_destination_segment_reports",
            tuple(segment_reports),
        )
        object.__setattr__(
            self,
            "_destination_expanded_output_slices",
            tuple(expanded_output_slices),
        )

    def _destination_native_arguments(self):
        return (
            self.destination_segment_source_offsets,
            self.destination_segment_input_offsets,
            self.destination_segment_input_dimensions,
            self.destination_segment_workspace_offsets,
            self.destination_segment_workspace_dimensions,
            self.destination_segment_node_offsets,
            self.destination_segment_root_offsets,
            self.destination_segment_projection_offsets,
            self.destination_segment_projection_dimensions,
            self.destination_segment_output_offsets,
            self.destination_node_offsets,
            self.destination_node_dimensions,
            self.destination_node_leaf_offsets,
            self.destination_node_left,
            self.destination_node_right,
            self.destination_node_coefficient_offsets,
            self.destination_coefficient_rows,
            self.destination_coefficient_columns,
            self.destination_coefficient_values,
            self.destination_root_nodes,
            self.destination_root_projection_starts,
            self.destination_root_projection_dimensions,
            self.destination_root_output_offsets,
            self.destination_projection_values,
            self.destination_total_source_count,
            self.destination_workspace_dimension,
            self.destination_output_dimension,
        )

    def _destination_segmented_output(self, unique_powers, leading_shape):
        batch_size = 1
        for size in leading_shape:
            batch_size *= int(size)
        flat_powers = torch.cat(
            tuple(
                value.reshape(int(batch_size), -1)
                for value in unique_powers
            ),
            dim=1,
        )
        if int(flat_powers.shape[1]) != int(self.power_output_dimension):
            raise RuntimeError(
                "hierarchical power arena width is inconsistent"
            )
        padded = torch.cat(
            (
                flat_powers,
                flat_powers.new_zeros((int(batch_size), 1)),
            ),
            dim=1,
        )
        packed_slots = padded.index_select(
            1,
            self.destination_power_gather_indices,
        ).contiguous()
        if int(packed_slots.shape[1]) != int(
            self.destination_input_dimension
        ):
            raise RuntimeError(
                "hierarchical destination input width is inconsistent"
            )
        expanded = torch.ops.ye3t_runtime.factorized_angular_segmented(
            packed_slots,
            *self._destination_native_arguments(),
        )
        reduced = tuple(
            expanded[:, int(record["start"]):int(record["stop"])].reshape(
                int(batch_size),
                int(record["root_count"]),
                int(record["output_dimension"]),
            ).sum(dim=1)
            for record in self._destination_expanded_output_slices
        )
        return torch.cat(reduced, dim=1)

    def _evaluate_unique_powers(self, block_inputs):
        block_inputs = tuple(block_inputs)
        if len(block_inputs) != int(self.unique_binding_count):
            raise ValueError(
                "hierarchical grouped input count does not match physical bindings"
            )
        evaluated = [None] * len(self._unique_power_records)
        leading_shape = None
        first = self._source_modules[0]
        if self.block_power_execution_policy == "shared_monomial":
            for group in self._shared_power_groups:
                value = torch.as_tensor(
                    block_inputs[int(group["binding_index"])],
                    dtype=first.lr_synthesis.dtype,
                    device=first.lr_synthesis.device,
                )
                if int(value.shape[-1]) != int(group["input_dimension"]):
                    raise ValueError(
                        "hierarchical shared-power input width is incorrect"
                    )
                current_leading = tuple(value.shape[:-1])
                if leading_shape is None:
                    leading_shape = current_leading
                elif current_leading != leading_shape:
                    raise ValueError(
                        "hierarchical grouped inputs require matching "
                        "leading shapes"
                    )
                packed = symmetric_power_shared_monomial_contraction(
                    value,
                    *tuple(
                        getattr(self, name)
                        for name in group["buffer_names"]
                    ),
                    backend=(
                        "native"
                        if first.backend in {"auto", "native"}
                        else "reference"
                    ),
                )
                for output_slice in group["output_slices"]:
                    power_index = int(output_slice["power_index"])
                    record = self._unique_power_records[power_index]
                    magnetic_count = 2 * int(record["output_L"]) + 1
                    plan = record["plan"]
                    evaluated[power_index] = packed[
                        ...,
                        int(output_slice["start"]):int(output_slice["stop"]),
                    ].reshape(
                        current_leading
                        + (
                            int(plan.descriptor_count) // magnetic_count,
                            magnetic_count,
                        )
                    )
            if any(value is None for value in evaluated):
                raise RuntimeError(
                    "hierarchical shared-power schedule omitted a request"
                )
            return tuple(evaluated), tuple(leading_shape)
        for power_index, record in enumerate(self._unique_power_records):
            value = torch.as_tensor(
                block_inputs[int(record["binding_index"])],
                dtype=first.lr_synthesis.dtype,
                device=first.lr_synthesis.device,
            )
            plan = record["plan"]
            if int(value.shape[-1]) != int(plan.channel_count):
                raise ValueError(
                    "hierarchical grouped physical multiplet width is incorrect"
                )
            current_leading = tuple(value.shape[:-1])
            if leading_shape is None:
                leading_shape = current_leading
            elif current_leading != leading_shape:
                raise ValueError(
                    "hierarchical grouped inputs require matching leading shapes"
                )
            result = symmetric_power_product_plan_contraction(
                value,
                plan,
                backend=first.backend,
            )
            magnetic_count = 2 * int(record["output_L"]) + 1
            if int(plan.descriptor_count) % magnetic_count:
                raise ValueError(
                    "hierarchical grouped power output is not multiplet-complete"
                )
            evaluated[power_index] = (
                result.reshape(
                    current_leading
                    + (int(plan.descriptor_count) // magnetic_count, magnetic_count)
                )
            )
        return tuple(evaluated), tuple(leading_shape)

    def _logical_outputs(self, block_inputs):
        unique_powers, leading_shape = self._evaluate_unique_powers(block_inputs)
        if self.destination_native_enabled:
            packed = self._destination_segmented_output(
                unique_powers,
                leading_shape,
            )
            outputs = tuple(
                packed[:, int(record["start"]):int(record["stop"])].reshape(
                    leading_shape
                    + (
                        int(module.channel_count),
                        int(module.tableau_count),
                        int(module.magnetic_count),
                    )
                )
                for record, module in zip(
                    self.output_slices,
                    self._source_modules,
                )
            )
            self.last_backend = (
                "shared_block_powers_then_destination_segmented_schur_lr"
            )
            return outputs
        outputs = []
        for module, power_indices in zip(
            self._source_modules,
            self._module_power_indices,
        ):
            evaluated = {
                key: unique_powers[int(power_index)]
                for key, power_index in power_indices.items()
            }
            outputs.append(
                module._forward_from_evaluated(evaluated, leading_shape)
            )
        self.last_backend = "shared_block_powers_then_compiler_parent_schur_lr"
        return tuple(outputs)

    def forward(self, block_inputs):
        return self._logical_outputs(block_inputs)

    def forward_packed(self, block_inputs):
        if self.destination_native_enabled:
            unique_powers, leading_shape = self._evaluate_unique_powers(
                block_inputs
            )
            output = self._destination_segmented_output(
                unique_powers,
                leading_shape,
            )
            self.last_backend = (
                "shared_block_powers_then_destination_segmented_schur_lr"
            )
            return output
        outputs = self._logical_outputs(block_inputs)
        leading_shape = tuple(outputs[0].shape[:-3])
        batch_size = 1
        for size in leading_shape:
            batch_size *= int(size)
        return torch.cat(
            tuple(
                output.reshape(int(batch_size), int(module.output_dimension))
                for output, module in zip(outputs, self._source_modules)
            ),
            dim=1,
        )

    def runtime_report(self):
        return {
            "runtime": "YE3THierarchicalRepeatedBlockModuleGroup",
            "member_plan_count": int(len(self._source_modules)),
            "member_plan_hashes": tuple(
                str(module.plan_hash) for module in self._source_modules
            ),
            "unique_physical_binding_count": int(self.unique_binding_count),
            "block_power_request_count": int(self.power_request_count),
            "block_power_evaluation_count": int(self.power_evaluation_count),
            "block_power_reuse_count": int(
                self.power_request_count - self.power_evaluation_count
            ),
            "block_power_execution_policy": str(
                self.block_power_execution_policy
            ),
            "shared_power_group_count": int(
                self.shared_power_group_count
            ),
            "block_power_runtime_call_count": int(
                self.shared_power_group_count
                if self.block_power_execution_policy == "shared_monomial"
                else self.power_evaluation_count
            ),
            "shared_power_groups": tuple(
                {
                    "binding_index": int(record["binding_index"]),
                    "power_indices": tuple(record["power_indices"]),
                    "input_dimension": int(record["input_dimension"]),
                    "output_dimension": int(record["output_dimension"]),
                    "coefficient_count": int(
                        record["coefficient_count"]
                    ),
                    "unique_monomial_count": int(
                        record["unique_monomial_count"]
                    ),
                }
                for record in getattr(self, "_shared_power_groups", ())
            ),
            "block_power_table_reports": tuple(
                {
                    "binding_index": int(record["binding_index"]),
                    "output_L": int(record["output_L"]),
                    **symmetric_power_product_plan_table_report(
                        record["plan"],
                        device=self._source_modules[0].lr_synthesis.device,
                        dtype=self._source_modules[0].lr_synthesis.dtype,
                    ),
                }
                for record in self._unique_power_records
            ),
            "output_dimension": int(self.output_dimension),
            "parent_execution_policy_requested": str(
                self.parent_execution_policy_requested
            ),
            "destination_native_capable": bool(
                self.destination_native_capable
            ),
            "destination_native_enabled": bool(
                self.destination_native_enabled
            ),
            "destination_native_call_count": (
                1 if self.destination_native_enabled else 0
            ),
            "destination_power_output_dimension": int(
                getattr(self, "power_output_dimension", 0)
            ),
            "destination_input_dimension": int(
                getattr(self, "destination_input_dimension", 0)
            ),
            "destination_workspace_dimension": int(
                getattr(self, "destination_workspace_dimension", 0)
            ),
            "destination_expanded_output_dimension": int(
                getattr(self, "destination_output_dimension", 0)
            ),
            "destination_post_native_reduction_count": (
                len(self._source_modules)
                if self.destination_native_enabled
                else 0
            ),
            "destination_segment_reports": tuple(
                dict(record)
                for record in getattr(
                    self,
                    "_destination_segment_reports",
                    (),
                )
            ),
            "destination_order": "compiler_module_order",
            "runtime_path_discovery": False,
            "supports_autograd_vjp_hvp": True,
            "last_backend": self.last_backend,
        }


__all__ = [
    "YE3THierarchicalRepeatedBlockModule",
    "YE3THierarchicalRepeatedBlockModuleGroup",
]
