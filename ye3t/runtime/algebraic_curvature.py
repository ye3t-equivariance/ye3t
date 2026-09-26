"""Derivative-complete Torch runtime for exact algebraic-curvature outputs."""

from collections.abc import Mapping
import hashlib
import numpy as np
import torch

from ye3t.couplings.algebraic_curvature import (
    AlgebraicCurvatureOutputPlan,
    AlgebraicCurvatureOutputSchedule,
    CompiledAlgebraicCurvatureOutput,
    materialize_algebraic_curvature_output_schedule,
)
from ye3t.couplings.algebraic_curvature_synthesis import CompiledAlgebraicCurvatureSynthesis


def _plan_and_schedule(value):
    if isinstance(value, CompiledAlgebraicCurvatureOutput):
        return value.plan, value.schedule
    if isinstance(value, AlgebraicCurvatureOutputSchedule):
        return value.plan, value
    if isinstance(value, AlgebraicCurvatureOutputPlan):
        return value, materialize_algebraic_curvature_output_schedule(value)
    raise TypeError("Expected an algebraic-curvature plan, schedule, or compiled output.")


def _real_tesseral_o3_action(angular_L, parity, matrix, *, dtype, device):
    from ye3t.representations.generalized_irreps import (
        AngularIrrep,
        CoupledIrrepLabel,
        PermutationIrrep,
        PermutationSubgroup,
        PermutationSubgroupFactor,
    )
    from ye3t.runtime.generalized import GeneralizedIrreps

    subgroup = PermutationSubgroup(
        (PermutationSubgroupFactor("binding_o3", 0, 1),)
    )
    permutation = PermutationIrrep.trivial_for_subgroup(subgroup)
    label = CoupledIrrepLabel(
        AngularIrrep(int(angular_L)),
        permutation,
        0,
        parity,
    )
    return GeneralizedIrreps(((1, label),)).D_from_matrix_real(
        matrix,
        dtype=dtype,
        device=device,
    )


def _schedule_array_hash(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()[:16]


def _compact_signed_permutation_schedule(
    free_rows,
    free_cols,
    quadruple_rows,
    quadruple_cols,
    pair_source_indices,
    pair_signs,
):
    """Compile the signed-pair action directly in the compact chart."""

    free_rows = np.asarray(free_rows, dtype=np.int64)
    free_cols = np.asarray(free_cols, dtype=np.int64)
    quadruple_rows = np.asarray(quadruple_rows, dtype=np.int64)
    quadruple_cols = np.asarray(quadruple_cols, dtype=np.int64)
    pair_source_indices = np.asarray(pair_source_indices, dtype=np.int64)
    pair_signs = np.asarray(pair_signs, dtype=np.float64)
    pair_dim = int(pair_source_indices.size)
    if pair_signs.shape != (pair_dim,):
        raise ValueError("pair_signs must match pair_source_indices.")
    if tuple(np.sort(pair_source_indices)) != tuple(range(pair_dim)):
        raise ValueError("pair_source_indices must be a complete permutation.")
    if not np.all(np.isfinite(pair_signs)) or not np.allclose(
        np.abs(pair_signs), 1.0, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("pair_signs must contain finite signed-unit phases.")
    if free_rows.shape != free_cols.shape:
        raise ValueError("Free-coordinate row and column schedules must match.")
    if quadruple_rows.shape != quadruple_cols.shape or (
        quadruple_rows.ndim != 2 or quadruple_rows.shape[1] != 3
    ):
        raise ValueError("Quadruple schedules must share shape (count,3).")

    free_lookup = np.full((pair_dim, pair_dim), -1, dtype=np.int32)
    free_indices = np.arange(free_rows.size, dtype=np.int32)
    free_lookup[free_rows, free_cols] = free_indices
    free_lookup[free_cols, free_rows] = free_indices
    source_free_rows = pair_source_indices[free_rows]
    source_free_cols = pair_source_indices[free_cols]
    free_source_indices = free_lookup[source_free_rows, source_free_cols]
    if np.any(free_source_indices < 0):
        raise ValueError(
            "The signed pair permutation does not preserve the free compact coordinates."
        )
    if np.unique(free_source_indices).size != free_rows.size:
        raise ValueError("The signed pair action is not bijective on free coordinates.")
    free_phases = pair_signs[free_rows] * pair_signs[free_cols]
    del free_lookup

    quadruple_count = int(quadruple_rows.shape[0])
    if not quadruple_count:
        quadruple_source_indices = np.empty(0, dtype=np.int64)
        quadruple_actions = np.empty((0, 2, 2), dtype=np.float64)
        maximum_plane_residual = 0.0
        maximum_orthogonality_residual = 0.0
    else:
        quadruple_lookup = np.full((pair_dim, pair_dim), -1, dtype=np.int32)
        encoded = (
            3 * np.arange(quadruple_count, dtype=np.int32)[:, None]
            + np.arange(3, dtype=np.int32)[None, :]
        )
        quadruple_lookup[quadruple_rows, quadruple_cols] = encoded
        quadruple_lookup[quadruple_cols, quadruple_rows] = encoded
        source_rows = pair_source_indices[quadruple_rows]
        source_cols = pair_source_indices[quadruple_cols]
        source_codes = quadruple_lookup[source_rows, source_cols]
        del quadruple_lookup
        if np.any(source_codes < 0):
            raise ValueError(
                "The signed pair permutation does not preserve four-index blocks."
            )
        source_blocks = source_codes // 3
        if not np.all(source_blocks == source_blocks[:, :1]):
            raise ValueError(
                "A destination four-index block maps to multiple source blocks."
            )
        quadruple_source_indices = np.asarray(source_blocks[:, 0], dtype=np.int64)
        if np.unique(quadruple_source_indices).size != quadruple_count:
            raise ValueError("The signed pair action is not bijective on four-index blocks.")
        source_pairings = source_codes % 3
        pairing_phases = (
            pair_signs[quadruple_rows] * pair_signs[quadruple_cols]
        )
        root_three = np.sqrt(3.0)
        unpack_plane = np.asarray(
            (
                (0.5, -1.0 / np.sqrt(12.0)),
                (0.5, 1.0 / np.sqrt(12.0)),
                (0.0, 1.0 / root_three),
            ),
            dtype=np.float64,
        )
        pack_plane = np.asarray(
            (
                (1.0, 1.0, 0.0),
                (-1.0 / root_three, 1.0 / root_three, 2.0 / root_three),
            ),
            dtype=np.float64,
        )
        transformed_plane = (
            pairing_phases[..., None] * unpack_plane[source_pairings]
        )
        quadruple_actions = np.einsum(
            "ij,qjk->qik", pack_plane, transformed_plane
        )
        bianchi_normal = np.asarray((1.0, -1.0, 1.0), dtype=np.float64)
        maximum_plane_residual = float(
            np.max(np.abs(np.einsum("j,qjk->qk", bianchi_normal, transformed_plane)))
        )
        identity = np.eye(2, dtype=np.float64)
        maximum_orthogonality_residual = float(
            np.max(
                np.abs(
                    np.einsum("qij,qkj->qik", quadruple_actions, quadruple_actions)
                    - identity
                )
            )
        )
        if maximum_plane_residual > 2.0e-12:
            raise ValueError(
                "The signed pair action does not preserve the Bianchi plane."
            )
        if maximum_orthogonality_residual > 2.0e-12:
            raise ValueError(
                "The compact four-index action is not orthogonal."
            )

    schedule_hash = _schedule_array_hash(
        (
            np.asarray(free_source_indices, dtype=np.int64),
            np.asarray(free_phases, dtype=np.float64),
            quadruple_source_indices,
            quadruple_actions,
        )
    )
    return {
        "free_source_indices": np.asarray(free_source_indices, dtype=np.int64),
        "free_phases": np.asarray(free_phases, dtype=np.float64),
        "quadruple_source_indices": quadruple_source_indices,
        "quadruple_actions": quadruple_actions,
        "schedule_hash": schedule_hash,
        "maximum_bianchi_plane_residual": maximum_plane_residual,
        "maximum_orthogonality_residual": maximum_orthogonality_residual,
    }


class YE3TAlgebraicCurvatureOutput(torch.nn.Module):
    """Pack and reconstruct the exact real ``S_(2,2)`` output chart.

    The module is linear.  Its registered index buffers are generated by
    ``ye3t.couplings`` and therefore carry compiler provenance; no Young path
    or coefficient is discovered at runtime.
    """

    def __init__(self, compiled_or_plan):
        super().__init__()
        plan, schedule = _plan_and_schedule(compiled_or_plan)
        self.one_particle_dim = int(plan.one_particle_dim)
        self.pair_dim = int(plan.pair_dim)
        self.compact_dim = int(plan.compact_dim)
        self.free_coordinate_dim = int(plan.free_coordinate_dim)
        self.four_form_dim = int(plan.four_form_dim)
        self.convention_hash = str(plan.convention_hash)
        self.schedule_hash = str(schedule.schedule_hash)
        self.plan_payload = plan.to_dict()
        self._signed_permutation_metadata = {}
        self.register_buffer(
            "pair_basis",
            torch.as_tensor(
                np.asarray(plan.pair_basis_tuples, dtype=np.int64).reshape(-1, 2)
            ),
            persistent=True,
        )
        self.register_buffer(
            "free_rows",
            torch.as_tensor(np.asarray(schedule.free_rows, dtype=np.int64)),
            persistent=True,
        )
        self.register_buffer(
            "free_cols",
            torch.as_tensor(np.asarray(schedule.free_cols, dtype=np.int64)),
            persistent=True,
        )
        self.register_buffer(
            "quadruple_rows",
            torch.as_tensor(np.asarray(schedule.quadruple_rows, dtype=np.int64)),
            persistent=True,
        )
        self.register_buffer(
            "quadruple_cols",
            torch.as_tensor(np.asarray(schedule.quadruple_cols, dtype=np.int64)),
            persistent=True,
        )

    @staticmethod
    def _require_real(value):
        if torch.is_complex(value):
            raise ValueError("The current algebraic-curvature output plan is real-valued.")

    def unpack(self, compact):
        """Reconstruct a symmetric, Bianchi-exact wedge-pair matrix."""

        self._require_real(compact)
        if int(compact.shape[-1]) != self.compact_dim:
            raise ValueError(
                f"Expected compact dimension {self.compact_dim}, got {compact.shape[-1]}."
            )
        output = compact.new_zeros(compact.shape[:-1] + (self.pair_dim, self.pair_dim))
        free = compact[..., : self.free_coordinate_dim]
        diagonal = torch.arange(self.pair_dim, device=compact.device)
        output[..., diagonal, diagonal] = free[..., : self.pair_dim]
        if self.free_coordinate_dim > self.pair_dim:
            rows = self.free_rows[self.pair_dim :]
            cols = self.free_cols[self.pair_dim :]
            shared = free[..., self.pair_dim :] / compact.new_tensor(2.0).sqrt()
            output[..., rows, cols] = shared
            output[..., cols, rows] = shared
        if self.four_form_dim:
            quad = compact[..., self.free_coordinate_dim :].reshape(
                compact.shape[:-1] + (self.four_form_dim, 2)
            )
            first = quad[..., 0]
            second = quad[..., 1]
            values = (
                0.5 * first - second / compact.new_tensor(12.0).sqrt(),
                0.5 * first + second / compact.new_tensor(12.0).sqrt(),
                second / compact.new_tensor(3.0).sqrt(),
            )
            for pairing in range(3):
                rows = self.quadruple_rows[:, pairing]
                cols = self.quadruple_cols[:, pairing]
                output[..., rows, cols] = values[pairing]
                output[..., cols, rows] = values[pairing]
        return output

    def pack(self, matrix):
        """Apply the Frobenius adjoint of ``unpack``."""

        self._require_real(matrix)
        expected = (self.pair_dim, self.pair_dim)
        if tuple(matrix.shape[-2:]) != expected:
            raise ValueError(f"Expected trailing matrix shape {expected}, got {tuple(matrix.shape[-2:])}.")
        diagonal = torch.arange(self.pair_dim, device=matrix.device)
        free_parts = [matrix[..., diagonal, diagonal]]
        if self.free_coordinate_dim > self.pair_dim:
            rows = self.free_rows[self.pair_dim :]
            cols = self.free_cols[self.pair_dim :]
            shared = (matrix[..., rows, cols] + matrix[..., cols, rows]) / matrix.new_tensor(2.0).sqrt()
            free_parts.append(shared)
        free = torch.cat(free_parts, dim=-1)
        if not self.four_form_dim:
            return free
        z_values = []
        for pairing in range(3):
            rows = self.quadruple_rows[:, pairing]
            cols = self.quadruple_cols[:, pairing]
            z_values.append(
                (matrix[..., rows, cols] + matrix[..., cols, rows])
                / matrix.new_tensor(2.0).sqrt()
            )
        first = (z_values[0] + z_values[1]) / matrix.new_tensor(2.0).sqrt()
        second = (
            -z_values[0] + z_values[1] + 2.0 * z_values[2]
        ) / matrix.new_tensor(6.0).sqrt()
        quad = torch.stack((first, second), dim=-1).reshape(matrix.shape[:-2] + (-1,))
        return torch.cat((free, quad), dim=-1)

    def project(self, matrix):
        """Orthogonally project a real wedge matrix into ``S_(2,2)``."""

        return self.unpack(self.pack(matrix))

    def register_signed_permutation(
        self,
        name,
        pair_source_indices,
        pair_signs,
        *,
        dtype=None,
        device=None,
    ):
        """Compile and register one exact compact signed-pair action."""

        name = str(name)
        if not name or not all(character.isalnum() or character == "_" for character in name):
            raise ValueError("Signed-permutation names must be nonempty alphanumeric identifiers.")
        if name in self._signed_permutation_metadata:
            raise ValueError(f"Signed permutation {name!r} is already registered.")
        schedule = _compact_signed_permutation_schedule(
            self.free_rows.detach().cpu().numpy(),
            self.free_cols.detach().cpu().numpy(),
            self.quadruple_rows.detach().cpu().numpy(),
            self.quadruple_cols.detach().cpu().numpy(),
            pair_source_indices,
            pair_signs,
        )
        floating_dtype = torch.get_default_dtype() if dtype is None else dtype
        buffers = {
            "free_source_indices": torch.as_tensor(
                schedule["free_source_indices"], dtype=torch.long, device=device
            ),
            "free_phases": torch.as_tensor(
                schedule["free_phases"], dtype=floating_dtype, device=device
            ),
            "quadruple_source_indices": torch.as_tensor(
                schedule["quadruple_source_indices"], dtype=torch.long, device=device
            ),
            "quadruple_actions": torch.as_tensor(
                schedule["quadruple_actions"], dtype=floating_dtype, device=device
            ),
        }
        buffer_names = {}
        for key, value in buffers.items():
            buffer_name = f"signed_permutation_{name}_{key}"
            self.register_buffer(buffer_name, value, persistent=True)
            buffer_names[key] = buffer_name
        self._signed_permutation_metadata[name] = {
            "schedule_hash": str(schedule["schedule_hash"]),
            "maximum_bianchi_plane_residual": float(
                schedule["maximum_bianchi_plane_residual"]
            ),
            "maximum_orthogonality_residual": float(
                schedule["maximum_orthogonality_residual"]
            ),
            "buffers": buffer_names,
        }
        return dict(self._signed_permutation_metadata[name])

    def apply_signed_permutation(self, compact, name):
        """Apply a registered signed-pair action without dense reconstruction."""

        self._require_real(compact)
        if int(compact.shape[-1]) != self.compact_dim:
            raise ValueError(
                f"Expected compact dimension {self.compact_dim}, got {compact.shape[-1]}."
            )
        name = str(name)
        if name not in self._signed_permutation_metadata:
            raise ValueError(f"Unknown signed permutation {name!r}.")
        metadata = self._signed_permutation_metadata[name]
        free_indices = getattr(self, metadata["buffers"]["free_source_indices"])
        free_phases = getattr(self, metadata["buffers"]["free_phases"])
        if compact.dtype != free_phases.dtype or compact.device != free_phases.device:
            raise ValueError(
                "Compact coordinates must match the signed-permutation dtype and device."
            )
        free = compact[..., : self.free_coordinate_dim].index_select(
            -1, free_indices
        ) * free_phases
        if not self.four_form_dim:
            return free
        quadruple_indices = getattr(
            self, metadata["buffers"]["quadruple_source_indices"]
        )
        actions = getattr(self, metadata["buffers"]["quadruple_actions"])
        quadruple = compact[..., self.free_coordinate_dim :].reshape(
            compact.shape[:-1] + (self.four_form_dim, 2)
        ).index_select(-2, quadruple_indices)
        transformed = torch.einsum("qij,...qj->...qi", actions, quadruple)
        return torch.cat((free, transformed.reshape(compact.shape[:-1] + (-1,))), dim=-1)

    def _exterior_density_entries(self, density, rows, cols):
        row_pairs = self.pair_basis.index_select(0, rows)
        col_pairs = self.pair_basis.index_select(0, cols)
        i = row_pairs[:, 0]
        j = row_pairs[:, 1]
        k = col_pairs[:, 0]
        l = col_pairs[:, 1]
        return (
            density[..., i, k] * density[..., j, l]
            - density[..., i, l] * density[..., j, k]
        )

    def compact_exterior_density(self, one_particle_density):
        r"""Return ``pack(Lambda^2 P)`` without a dense wedge matrix."""

        self._require_real(one_particle_density)
        expected = (self.one_particle_dim, self.one_particle_dim)
        if tuple(one_particle_density.shape[-2:]) != expected:
            raise ValueError(
                f"Expected trailing density shape {expected}, got "
                f"{tuple(one_particle_density.shape[-2:])}."
            )
        if one_particle_density.device != self.pair_basis.device:
            raise ValueError("Density and algebraic-curvature schedules must share a device.")
        diagonal = torch.arange(self.pair_dim, device=one_particle_density.device)
        free_parts = [self._exterior_density_entries(one_particle_density, diagonal, diagonal)]
        if self.free_coordinate_dim > self.pair_dim:
            rows = self.free_rows[self.pair_dim :]
            cols = self.free_cols[self.pair_dim :]
            shared = (
                self._exterior_density_entries(one_particle_density, rows, cols)
                + self._exterior_density_entries(one_particle_density, cols, rows)
            ) / one_particle_density.new_tensor(2.0).sqrt()
            free_parts.append(shared)
        free = torch.cat(free_parts, dim=-1)
        if not self.four_form_dim:
            return free
        z_values = []
        for pairing in range(3):
            rows = self.quadruple_rows[:, pairing]
            cols = self.quadruple_cols[:, pairing]
            z_values.append(
                (
                    self._exterior_density_entries(one_particle_density, rows, cols)
                    + self._exterior_density_entries(one_particle_density, cols, rows)
                )
                / one_particle_density.new_tensor(2.0).sqrt()
            )
        first = (z_values[0] + z_values[1]) / one_particle_density.new_tensor(2.0).sqrt()
        second = (
            -z_values[0] + z_values[1] + 2.0 * z_values[2]
        ) / one_particle_density.new_tensor(6.0).sqrt()
        quadruple = torch.stack((first, second), dim=-1).reshape(
            one_particle_density.shape[:-2] + (-1,)
        )
        return torch.cat((free, quadruple), dim=-1)

    def bianchi_residual(self, matrix, floor=1.0e-30):
        """Return a relative Bianchi residual in the canonical pair chart."""

        self._require_real(matrix)
        if not self.four_form_dim:
            return matrix.new_zeros(matrix.shape[:-2])
        values = []
        for pairing in range(3):
            rows = self.quadruple_rows[:, pairing]
            cols = self.quadruple_cols[:, pairing]
            values.append(0.5 * (matrix[..., rows, cols] + matrix[..., cols, rows]))
        residual = values[0] - values[1] + values[2]
        numerator = torch.linalg.vector_norm(residual, dim=-1)
        denominator = torch.clamp(
            torch.linalg.vector_norm(matrix.reshape(matrix.shape[:-2] + (-1,)), dim=-1),
            min=matrix.new_tensor(float(floor)),
        )
        return numerator / denominator

    def hermiticity_residual(self, matrix, floor=1.0e-30):
        """Return the relative transpose-symmetry residual for the real plan."""

        numerator = torch.linalg.vector_norm(
            (matrix - matrix.transpose(-1, -2)).reshape(matrix.shape[:-2] + (-1,)),
            dim=-1,
        )
        denominator = torch.clamp(
            torch.linalg.vector_norm(matrix.reshape(matrix.shape[:-2] + (-1,)), dim=-1),
            min=matrix.new_tensor(float(floor)),
        )
        return numerator / denominator

    def forward(self, compact):
        return self.unpack(compact)

    def get_extra_state(self):
        return {
            "convention_hash": self.convention_hash,
            "schedule_hash": self.schedule_hash,
            "plan": dict(self.plan_payload),
            "signed_permutations": {
                key: {
                    item_key: item_value
                    for item_key, item_value in value.items()
                    if item_key != "buffers"
                }
                for key, value in self._signed_permutation_metadata.items()
            },
        }

    def set_extra_state(self, state):
        if str(state.get("convention_hash", "")) != self.convention_hash:
            raise ValueError("Serialized algebraic-curvature convention hash does not match the module plan.")
        if str(state.get("schedule_hash", "")) != self.schedule_hash:
            raise ValueError("Serialized algebraic-curvature schedule hash does not match the module plan.")
        expected_signed = {
            key: {
                item_key: item_value
                for item_key, item_value in value.items()
                if item_key != "buffers"
            }
            for key, value in self._signed_permutation_metadata.items()
        }
        saved_signed = dict(state.get("signed_permutations", {}))
        if saved_signed != expected_signed:
            raise ValueError(
                "Serialized algebraic-curvature signed-permutation schedules do not match."
            )
        self.plan_payload = dict(state.get("plan", self.plan_payload))

    def backend_report(self):
        return {
            "runtime": "YE3TAlgebraicCurvatureOutput",
            "compiler_owner": "ye3t.couplings",
            "one_particle_dim": self.one_particle_dim,
            "pair_dim": self.pair_dim,
            "compact_dim": self.compact_dim,
            "young_partition": (2, 2),
            "convention_hash": self.convention_hash,
            "schedule_hash": self.schedule_hash,
            "forward": "implicit_frobenius_orthonormal_bianchi_chart",
            "adjoint": "exact_chart_transpose",
            "double_adjoint": "autograd_through_linear_chart",
            "ambient_projector_materialized": False,
            "psd_enforced": False,
            "signed_permutations": {
                key: {
                    item_key: item_value
                    for item_key, item_value in value.items()
                    if item_key != "buffers"
                }
                for key, value in self._signed_permutation_metadata.items()
            },
            "compact_exterior_density": "direct_schedule_no_dense_wedge_matrix",
        }


def _sector_name(L, parity):
    parity = str(parity)
    if parity not in {"even", "odd"}:
        raise ValueError("sector parity must be 'even' or 'odd'.")
    return f"L{int(L)}_{parity}"


def _geometry_multiplicity_map(value):
    if not isinstance(value, Mapping):
        raise TypeError("geometry_multiplicities must be a mapping keyed by (L, parity) or 'L#_parity'.")
    resolved = {}
    for key, multiplicity in value.items():
        if isinstance(key, tuple) and len(key) == 2:
            name = _sector_name(int(key[0]), str(key[1]))
        else:
            name = str(key)
        multiplicity = int(multiplicity)
        if multiplicity < 1:
            raise ValueError("Every requested geometry multiplicity must be positive.")
        if name in resolved:
            raise ValueError(f"Duplicate geometry multiplicity for sector {name!r}.")
        resolved[name] = multiplicity
    return resolved


def _binding_geometry_channel_map(value, required):
    required = tuple(required)
    if isinstance(value, bool):
        raise TypeError("geometry_channels must be a positive integer or a mapping.")
    if isinstance(value, int):
        if int(value) < 1:
            raise ValueError("geometry_channels must be positive.")
        return {key: int(value) for key in required}
    if not isinstance(value, Mapping):
        raise TypeError(
            "geometry_channels must be a positive integer or a mapping keyed by "
            "(binding_type_id,L,parity)."
        )
    resolved = {}
    for key, channels in value.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError(
                "Mapped geometry_channels keys must be "
                "(binding_type_id,L,parity) tuples."
            )
        canonical = (str(key[0]), int(key[1]), str(key[2]))
        _sector_name(canonical[1], canonical[2])
        channels = int(channels)
        if channels < 1:
            raise ValueError("Every geometry channel count must be positive.")
        if canonical in resolved:
            raise ValueError(f"Duplicate geometry channel count for {canonical!r}.")
        resolved[canonical] = channels
    missing = sorted(set(required).difference(resolved))
    extra = sorted(set(resolved).difference(required))
    if missing or extra:
        raise ValueError(
            "geometry_channels must match every binding-type output sector exactly; "
            f"missing={missing!r}, extra={extra!r}."
        )
    return resolved


class YE3TAlgebraicCurvatureReadout(torch.nn.Module):
    r"""Synthesize a physical ``(2,2)`` operator from complete carriers.

    The fixed compiler basis and the only learned map are

    .. math::

       A_o(X)=\sum_{L,m,\alpha,\beta}
       S^L_{o;\alpha m} W^L_{\alpha\beta}G^L_{\beta m}(X).

    ``W^L`` has exactly two axes: output multiplicity and geometry channel.
    It has no magnetic axis.  Inputs must therefore provide every component
    ``m=0..2L`` of each declared real-tesseral geometry carrier.
    """

    def __init__(
        self,
        compiled,
        geometry_multiplicities,
        *,
        dtype=None,
        device=None,
    ):
        super().__init__()
        if not isinstance(compiled, CompiledAlgebraicCurvatureSynthesis):
            raise TypeError("compiled must be a CompiledAlgebraicCurvatureSynthesis.")
        if not bool(compiled.validation_report.get("passed", False)):
            raise ValueError("The physical-output synthesis compiler certificate did not pass.")
        self.output = YE3TAlgebraicCurvatureOutput(compiled.plan.output)
        if device is not None:
            self.output.to(device=device)
        self.convention_hash = str(compiled.plan.convention_hash)
        self.coefficient_hash = str(compiled.coefficient_hash)
        self.compiler_payload = compiled.to_dict()
        self.sector_inventory = tuple(dict(item) for item in compiled.sector_inventory)
        geometry_map = _geometry_multiplicity_map(geometry_multiplicities)
        required = {
            _sector_name(item["L"], item["parity"])
            for item in self.sector_inventory
        }
        missing = sorted(required.difference(geometry_map))
        extra = sorted(set(geometry_map).difference(required))
        if missing or extra:
            raise ValueError(
                "geometry_multiplicities must match every compiled output sector exactly; "
                f"missing={missing!r}, extra={extra!r}."
            )
        self.geometry_multiplicities = dict(geometry_map)
        self.weights = torch.nn.ParameterDict()
        parameter_dtype = torch.get_default_dtype() if dtype is None else dtype
        for item in self.sector_inventory:
            name = _sector_name(item["L"], item["parity"])
            weight = torch.empty(
                int(item["output_multiplicity"]),
                int(geometry_map[name]),
                dtype=parameter_dtype,
                device=device,
            )
            torch.nn.init.xavier_uniform_(weight)
            self.weights[name] = torch.nn.Parameter(weight)

        self._template_sector_buffers = {}
        for template_index, template in enumerate(tuple(compiled.templates)):
            for sector in tuple(template.sectors):
                name = _sector_name(sector.L, sector.parity)
                buffer_name = f"template_{template_index}_{name}_basis"
                self.register_buffer(
                    buffer_name,
                    torch.as_tensor(np.asarray(sector.basis, dtype=np.float64), dtype=parameter_dtype, device=device),
                    persistent=True,
                )
                self._template_sector_buffers[(int(template_index), name)] = buffer_name

        embedding_local = []
        embedding_global = []
        embedding_values = []
        binding_runtime = []
        cursor = 0
        for binding in tuple(compiled.bindings):
            count = int(binding.embedding_values.size)
            embedding_local.append(np.asarray(binding.embedding_local_rows, dtype=np.int64))
            embedding_global.append(np.asarray(binding.embedding_global_rows, dtype=np.int64))
            embedding_values.append(np.asarray(binding.embedding_values, dtype=np.float64))
            binding_runtime.append(
                {
                    "template_index": int(binding.template_index),
                    "embedding_start": int(cursor),
                    "embedding_stop": int(cursor + count),
                    "sector_offsets": tuple(dict(item) for item in binding.sector_offsets),
                }
            )
            cursor += count
        local_array = np.concatenate(embedding_local) if embedding_local else np.zeros(0, dtype=np.int64)
        global_array = np.concatenate(embedding_global) if embedding_global else np.zeros(0, dtype=np.int64)
        value_array = np.concatenate(embedding_values) if embedding_values else np.zeros(0, dtype=np.float64)
        self.register_buffer("embedding_local_rows", torch.as_tensor(local_array, dtype=torch.long, device=device), persistent=True)
        self.register_buffer("embedding_global_rows", torch.as_tensor(global_array, dtype=torch.long, device=device), persistent=True)
        self.register_buffer(
            "embedding_values",
            torch.as_tensor(value_array, dtype=parameter_dtype, device=device),
            persistent=True,
        )
        self._binding_runtime = tuple(binding_runtime)
        self._template_row_counts = tuple(int(len(template.content_compact_rows)) for template in compiled.templates)

    @property
    def compact_dim(self):
        return int(self.output.compact_dim)

    def _carrier(self, geometry_carriers, item):
        name = _sector_name(item["L"], item["parity"])
        tuple_key = (int(item["L"]), str(item["parity"]))
        if tuple_key in geometry_carriers:
            carrier = geometry_carriers[tuple_key]
        elif name in geometry_carriers:
            carrier = geometry_carriers[name]
        else:
            raise ValueError(f"Missing complete geometry carrier {name!r}.")
        if torch.is_complex(carrier):
            raise ValueError("The compiled physical-output synthesis basis is real-tesseral.")
        expected = (int(self.geometry_multiplicities[name]), 2 * int(item["L"]) + 1)
        if carrier.ndim < 2 or tuple(carrier.shape[-2:]) != expected:
            raise ValueError(
                f"Geometry carrier {name!r} requires trailing shape {expected}, "
                f"got {tuple(carrier.shape[-2:])}."
            )
        weight = self.weights[name]
        if carrier.dtype != weight.dtype or carrier.device != weight.device:
            raise ValueError(
                f"Geometry carrier {name!r} must match readout dtype/device "
                f"{weight.dtype}/{weight.device}."
            )
        return carrier

    def forward(self, geometry_carriers):
        """Return compact physical-output coordinates."""

        if not isinstance(geometry_carriers, Mapping):
            raise TypeError("geometry_carriers must be a mapping of complete sector tensors.")
        coefficients = {}
        batch_shape = None
        reference = None
        for item in self.sector_inventory:
            name = _sector_name(item["L"], item["parity"])
            carrier = self._carrier(geometry_carriers, item)
            current_batch = tuple(carrier.shape[:-2])
            if batch_shape is None:
                batch_shape = current_batch
                reference = carrier
            elif current_batch != batch_shape:
                raise ValueError("All geometry carriers must have identical batch dimensions.")
            coefficients[name] = torch.einsum("ab,...bm->...am", self.weights[name], carrier)
        compact = reference.new_zeros(batch_shape + (self.compact_dim,))
        for binding in self._binding_runtime:
            template_index = int(binding["template_index"])
            local = reference.new_zeros(batch_shape + (int(self._template_row_counts[template_index]),))
            for offset in tuple(binding["sector_offsets"]):
                name = _sector_name(offset["L"], offset["parity"])
                basis = getattr(self, self._template_sector_buffers[(template_index, name)])
                values = coefficients[name][..., int(offset["start"]):int(offset["stop"]), :]
                local = local + torch.einsum("ram,...am->...r", basis, values)
            start = int(binding["embedding_start"])
            stop = int(binding["embedding_stop"])
            local_rows = self.embedding_local_rows[start:stop]
            global_rows = self.embedding_global_rows[start:stop]
            values = self.embedding_values[start:stop]
            contribution = local.index_select(-1, local_rows) * values
            compact.index_add_(-1, global_rows, contribution)
        return compact

    def reconstruct(self, geometry_carriers):
        """Return the symmetric Bianchi-exact wedge matrix prediction."""

        return self.output.unpack(self.forward(geometry_carriers))

    def get_extra_state(self):
        return {
            "convention_hash": self.convention_hash,
            "coefficient_hash": self.coefficient_hash,
            "geometry_multiplicities": dict(self.geometry_multiplicities),
            "compiler": dict(self.compiler_payload),
        }

    def set_extra_state(self, state):
        if str(state.get("convention_hash", "")) != self.convention_hash:
            raise ValueError("Serialized synthesis convention hash does not match the compiled plan.")
        if str(state.get("coefficient_hash", "")) != self.coefficient_hash:
            raise ValueError("Serialized synthesis coefficient hash does not match the compiled plan.")
        if dict(state.get("geometry_multiplicities", {})) != self.geometry_multiplicities:
            raise ValueError("Serialized geometry multiplicities do not match the readout.")
        self.compiler_payload = dict(state.get("compiler", self.compiler_payload))

    def backend_report(self):
        return {
            "runtime": "YE3TAlgebraicCurvatureReadout",
            "compiler_owner": "ye3t.couplings.compile_algebraic_curvature_synthesis",
            "convention_hash": self.convention_hash,
            "coefficient_hash": self.coefficient_hash,
            "fixed_output_partition": (2, 2),
            "learned_axes": ("output_multiplicity", "geometry_channel"),
            "complete_axes": ("magnetic", "tableau"),
            "tableau_realization": "complete_physical_(2,2)_Bianchi_plane",
            "template_count": int(len(self._template_row_counts)),
            "binding_count": int(len(self._binding_runtime)),
            "compact_output_dim": int(self.compact_dim),
            "dense_global_basis_materialized": False,
            "dense_ambient_projector_materialized": False,
            "forward_backend": "torch_factorized_template_then_sparse_compact_scatter",
            "adjoint": "torch_autograd_exact_linear_transpose",
            "double_adjoint": "torch_autograd_exact_linear_map",
        }


class YE3TBindingAttachedAlgebraicCurvatureReadout(torch.nn.Module):
    r"""Synthesize an exact ``(2,2)`` operator from binding-attached carriers.

    For physical binding ``b`` of compiler-certified type ``tau``, this first
    conservative lowering evaluates

    .. math::

       c^L_{b\alpha m}
       =\sum_q w^L_{\tau q}G^L_{b\alpha q m},
       \qquad
       A_o=\sum_{b,L,\alpha,m}S^L_{o;b\alpha m}c^L_{b\alpha m}.

    The learned map is the identity on local output multiplicity ``alpha`` and
    magnetic coordinate ``m``.  It therefore commutes with every exact
    binding-basis action compiled for the direct source.  Only binding type and
    geometry-channel axes are learned; the physical binding, Young/tableau,
    output-multiplicity, and magnetic axes remain complete.
    """

    def __init__(
        self,
        compiled,
        binding_type_ids,
        geometry_channels,
        *,
        active_carrier_keys=None,
        dtype=None,
        device=None,
    ):
        super().__init__()
        if not isinstance(compiled, CompiledAlgebraicCurvatureSynthesis):
            raise TypeError("compiled must be a CompiledAlgebraicCurvatureSynthesis.")
        if not bool(compiled.validation_report.get("passed", False)):
            raise ValueError("The physical-output synthesis compiler certificate did not pass.")
        binding_type_ids = tuple(str(value) for value in tuple(binding_type_ids))
        if len(binding_type_ids) != len(compiled.bindings):
            raise ValueError(
                "binding_type_ids must provide exactly one type for every compiled binding."
            )
        if any(not value for value in binding_type_ids):
            raise ValueError("binding_type_ids may not contain empty identifiers.")

        self.output = YE3TAlgebraicCurvatureOutput(compiled.plan.output)
        if device is not None:
            self.output.to(device=device)
        self.convention_hash = str(compiled.plan.convention_hash)
        self.coefficient_hash = str(compiled.coefficient_hash)
        self.compiler_payload = compiled.to_dict()
        self.binding_type_ids = binding_type_ids
        parameter_dtype = torch.get_default_dtype() if dtype is None else dtype

        type_order = tuple(dict.fromkeys(binding_type_ids))
        type_binding_indices = {
            type_id: tuple(
                index
                for index, candidate in enumerate(binding_type_ids)
                if candidate == type_id
            )
            for type_id in type_order
        }
        type_templates = {}
        required = []
        for type_id in type_order:
            template_indices = {
                int(compiled.bindings[index].template_index)
                for index in type_binding_indices[type_id]
            }
            if len(template_indices) != 1:
                raise ValueError(
                    f"Binding type {type_id!r} maps to multiple exact output templates."
                )
            template_index = int(next(iter(template_indices)))
            type_templates[type_id] = template_index
            for sector in tuple(compiled.templates[template_index].sectors):
                required.append((type_id, int(sector.L), str(sector.parity)))
        complete_required = set(required)
        if active_carrier_keys is None:
            active_carrier_keys = complete_required
        else:
            active_carrier_keys = {
                (str(key[0]), int(key[1]), str(key[2]))
                for key in tuple(active_carrier_keys)
            }
            unknown = sorted(active_carrier_keys.difference(complete_required))
            if unknown:
                raise ValueError(
                    "active_carrier_keys contains carriers outside the compiled synthesis: "
                    f"{unknown!r}."
                )
            if not active_carrier_keys:
                raise ValueError("active_carrier_keys may not be empty.")
        required = tuple(key for key in required if key in active_carrier_keys)
        self.active_carrier_keys = tuple(sorted(active_carrier_keys))
        self.geometry_channels = _binding_geometry_channel_map(
            geometry_channels,
            required,
        )

        self.weights = torch.nn.ParameterDict()
        self._parameter_names = {}
        for type_index, type_id in enumerate(type_order):
            template = compiled.templates[type_templates[type_id]]
            for sector in tuple(template.sectors):
                key = (type_id, int(sector.L), str(sector.parity))
                if key not in active_carrier_keys:
                    continue
                name = f"type_{type_index}_{_sector_name(sector.L, sector.parity)}"
                channels = int(self.geometry_channels[key])
                weight = torch.empty(channels, dtype=parameter_dtype, device=device)
                torch.nn.init.normal_(weight, mean=0.0, std=channels ** -0.5)
                self.weights[name] = torch.nn.Parameter(weight)
                self._parameter_names[key] = name

        self._template_sector_buffers = {}
        for template_index, template in enumerate(tuple(compiled.templates)):
            for sector in tuple(template.sectors):
                sector_name = _sector_name(sector.L, sector.parity)
                buffer_name = f"template_{template_index}_{sector_name}_basis"
                self.register_buffer(
                    buffer_name,
                    torch.as_tensor(
                        np.asarray(sector.basis, dtype=np.float64),
                        dtype=parameter_dtype,
                        device=device,
                    ),
                    persistent=True,
                )
                self._template_sector_buffers[(int(template_index), sector_name)] = buffer_name

        runtime_types = []
        carrier_names = {}
        inactive_binding_count = 0
        for type_index, type_id in enumerate(type_order):
            binding_indices = type_binding_indices[type_id]
            template_index = int(type_templates[type_id])
            template = compiled.templates[template_index]
            if not tuple(template.sectors):
                if any(
                    int(compiled.bindings[index].embedding_values.size) != 0
                    for index in binding_indices
                ):
                    raise RuntimeError(
                        "A zero-sector binding type has a nonempty physical scatter."
                    )
                inactive_binding_count += int(len(binding_indices))
                continue
            row_count = int(len(template.content_compact_rows))
            flat_local_rows = []
            global_rows = []
            values = []
            for binding_position, binding_index in enumerate(binding_indices):
                binding = compiled.bindings[int(binding_index)]
                flat_local_rows.append(
                    np.asarray(binding.embedding_local_rows, dtype=np.int64)
                    + int(binding_position) * row_count
                )
                global_rows.append(
                    np.asarray(binding.embedding_global_rows, dtype=np.int64)
                )
                values.append(np.asarray(binding.embedding_values, dtype=np.float64))
            flat_local = (
                np.concatenate(flat_local_rows)
                if flat_local_rows
                else np.zeros(0, dtype=np.int64)
            )
            global_array = (
                np.concatenate(global_rows)
                if global_rows
                else np.zeros(0, dtype=np.int64)
            )
            value_array = (
                np.concatenate(values)
                if values
                else np.zeros(0, dtype=np.float64)
            )
            local_name = f"type_{type_index}_embedding_local_flat_rows"
            global_name = f"type_{type_index}_embedding_global_rows"
            value_name = f"type_{type_index}_embedding_values"
            self.register_buffer(
                local_name,
                torch.as_tensor(flat_local, dtype=torch.long, device=device),
                persistent=True,
            )
            self.register_buffer(
                global_name,
                torch.as_tensor(global_array, dtype=torch.long, device=device),
                persistent=True,
            )
            self.register_buffer(
                value_name,
                torch.as_tensor(value_array, dtype=parameter_dtype, device=device),
                persistent=True,
            )
            sectors = []
            for sector in tuple(template.sectors):
                key = (type_id, int(sector.L), str(sector.parity))
                if key not in active_carrier_keys:
                    continue
                public_name = f"{type_id}::{_sector_name(sector.L, sector.parity)}"
                if public_name in carrier_names:
                    raise ValueError("Binding type identifiers create duplicate carrier names.")
                carrier_names[public_name] = key
                sectors.append(
                    {
                        "L": int(sector.L),
                        "parity": str(sector.parity),
                        "multiplicity": int(sector.multiplicity),
                        "channels": int(self.geometry_channels[key]),
                        "basis_buffer": self._template_sector_buffers[
                            (template_index, _sector_name(sector.L, sector.parity))
                        ],
                        "parameter_name": self._parameter_names[key],
                    }
                )
            if not sectors:
                inactive_binding_count += int(len(binding_indices))
                continue
            runtime_types.append(
                {
                    "binding_type_id": type_id,
                    "template_index": template_index,
                    "binding_indices": binding_indices,
                    "binding_count": int(len(binding_indices)),
                    "local_row_count": row_count,
                    "sectors": tuple(sectors),
                    "embedding_local_buffer": local_name,
                    "embedding_global_buffer": global_name,
                    "embedding_value_buffer": value_name,
                }
            )
        self._runtime_types = tuple(runtime_types)
        grouped_types = {}
        for runtime_type in self._runtime_types:
            signature = (
                int(runtime_type["template_index"]),
                tuple(
                    (
                        int(sector["L"]),
                        str(sector["parity"]),
                        int(sector["multiplicity"]),
                        int(sector["channels"]),
                        str(sector["basis_buffer"]),
                    )
                    for sector in tuple(runtime_type["sectors"])
                ),
            )
            grouped_types.setdefault(signature, []).append(runtime_type)
        runtime_groups = []
        for group_index, signature in enumerate(sorted(grouped_types, key=repr)):
            members = tuple(grouped_types[signature])
            row_count = int(members[0]["local_row_count"])
            binding_offset = 0
            local_rows = []
            global_rows = []
            values = []
            for runtime_type in members:
                if int(runtime_type["local_row_count"]) != row_count:
                    raise RuntimeError("Grouped output templates changed local row count.")
                for binding_position, binding_index in enumerate(
                    tuple(runtime_type["binding_indices"])
                ):
                    binding = compiled.bindings[int(binding_index)]
                    local_rows.append(
                        np.asarray(binding.embedding_local_rows, dtype=np.int64)
                        + int(binding_offset + binding_position) * row_count
                    )
                    global_rows.append(
                        np.asarray(binding.embedding_global_rows, dtype=np.int64)
                    )
                    values.append(np.asarray(binding.embedding_values, dtype=np.float64))
                binding_offset += int(runtime_type["binding_count"])
            flat_local = (
                np.concatenate(local_rows) if local_rows else np.zeros(0, dtype=np.int64)
            )
            global_array = (
                np.concatenate(global_rows) if global_rows else np.zeros(0, dtype=np.int64)
            )
            value_array = (
                np.concatenate(values) if values else np.zeros(0, dtype=np.float64)
            )
            local_name = f"group_{group_index}_embedding_local_flat_rows"
            global_name = f"group_{group_index}_embedding_global_rows"
            value_name = f"group_{group_index}_embedding_values"
            self.register_buffer(
                local_name,
                torch.as_tensor(flat_local, dtype=torch.long, device=device),
                persistent=True,
            )
            self.register_buffer(
                global_name,
                torch.as_tensor(global_array, dtype=torch.long, device=device),
                persistent=True,
            )
            self.register_buffer(
                value_name,
                torch.as_tensor(value_array, dtype=parameter_dtype, device=device),
                persistent=True,
            )
            runtime_groups.append(
                {
                    "runtime_types": members,
                    "binding_count": int(binding_offset),
                    "local_row_count": int(row_count),
                    "embedding_local_buffer": local_name,
                    "embedding_global_buffer": global_name,
                    "embedding_value_buffer": value_name,
                }
            )
        self._runtime_groups = tuple(runtime_groups)
        self._carrier_names = dict(carrier_names)
        self._inactive_binding_count = int(inactive_binding_count)

    @property
    def compact_dim(self):
        return int(self.output.compact_dim)

    def _normalize_carriers(self, geometry_carriers):
        if not isinstance(geometry_carriers, Mapping):
            raise TypeError(
                "geometry_carriers must map (binding_type_id,L,parity) or the "
                "reported public carrier name to a complete tensor."
            )
        normalized = {}
        for key, value in geometry_carriers.items():
            if isinstance(key, tuple) and len(key) == 3:
                canonical = (str(key[0]), int(key[1]), str(key[2]))
            elif isinstance(key, str) and key in self._carrier_names:
                canonical = self._carrier_names[key]
            else:
                raise ValueError(f"Unknown binding-attached geometry carrier key {key!r}.")
            if canonical in normalized:
                raise ValueError(f"Duplicate binding-attached geometry carrier {canonical!r}.")
            normalized[canonical] = value
        required = set(self.geometry_channels)
        missing = sorted(required.difference(normalized))
        extra = sorted(set(normalized).difference(required))
        if missing or extra:
            raise ValueError(
                "geometry_carriers must provide every binding-type sector exactly; "
                f"missing={missing!r}, extra={extra!r}."
            )
        return normalized

    def _carrier(self, carriers, runtime_type, sector):
        key = (
            str(runtime_type["binding_type_id"]),
            int(sector["L"]),
            str(sector["parity"]),
        )
        carrier = carriers[key]
        if torch.is_complex(carrier):
            raise ValueError("The binding-attached synthesis basis is real-tesseral.")
        expected = (
            int(runtime_type["binding_count"]),
            int(sector["multiplicity"]),
            int(sector["channels"]),
            2 * int(sector["L"]) + 1,
        )
        if carrier.ndim < 4 or tuple(carrier.shape[-4:]) != expected:
            raise ValueError(
                f"Geometry carrier {key!r} requires trailing shape {expected}, "
                f"got {tuple(carrier.shape[-4:])}."
            )
        weight = self.weights[str(sector["parameter_name"])]
        if carrier.dtype != weight.dtype or carrier.device != weight.device:
            raise ValueError(
                f"Geometry carrier {key!r} must match readout dtype/device "
                f"{weight.dtype}/{weight.device}."
            )
        return carrier, weight

    def forward(self, geometry_carriers):
        """Return exact compact ``(2,2)`` coordinates."""

        carriers = self._normalize_carriers(geometry_carriers)
        compact = None
        batch_shape = None
        for runtime_group in self._runtime_groups:
            local = None
            members = tuple(runtime_group["runtime_types"])
            for sector_index in range(len(tuple(members[0]["sectors"]))):
                carrier_parts = []
                weight_parts = []
                basis = None
                for runtime_type in members:
                    sector = tuple(runtime_type["sectors"])[sector_index]
                    carrier, weight = self._carrier(carriers, runtime_type, sector)
                    current_batch = tuple(carrier.shape[:-4])
                    if batch_shape is None:
                        batch_shape = current_batch
                        compact = carrier.new_zeros(batch_shape + (self.compact_dim,))
                    elif current_batch != batch_shape:
                        raise ValueError(
                            "All geometry carriers must have identical batch dimensions."
                        )
                    carrier_parts.append(carrier)
                    weight_parts.append(
                        weight.unsqueeze(0).expand(
                            int(runtime_type["binding_count"]), -1
                        )
                    )
                    current_basis = getattr(self, str(sector["basis_buffer"]))
                    if basis is None:
                        basis = current_basis
                    elif tuple(current_basis.shape) != tuple(basis.shape):
                        raise RuntimeError("Grouped output sectors changed basis shape.")
                carrier = (
                    carrier_parts[0]
                    if len(carrier_parts) == 1
                    else torch.cat(carrier_parts, dim=-4)
                )
                weight = (
                    weight_parts[0]
                    if len(weight_parts) == 1
                    else torch.cat(weight_parts, dim=0)
                )
                amplitudes = torch.einsum("bq,...baqm->...bam", weight, carrier)
                contribution = torch.einsum("ram,...bam->...br", basis, amplitudes)
                local = contribution if local is None else local + contribution
            flat_local = local.reshape(
                batch_shape
                + (
                    int(runtime_group["binding_count"])
                    * int(runtime_group["local_row_count"]),
                )
            )
            local_rows = getattr(self, str(runtime_group["embedding_local_buffer"]))
            global_rows = getattr(self, str(runtime_group["embedding_global_buffer"]))
            values = getattr(self, str(runtime_group["embedding_value_buffer"]))
            scattered = flat_local.index_select(-1, local_rows) * values
            compact.index_add_(-1, global_rows, scattered)
        if compact is None:
            raise ValueError("The compiled binding-attached readout has no nonzero sectors.")
        return compact

    def reconstruct(self, geometry_carriers):
        """Return the symmetric, Bianchi-exact wedge operator."""

        return self.output.unpack(self.forward(geometry_carriers))

    def transform_geometry_carriers_by_o3(self, geometry_carriers, matrix):
        """Transform every complete magnetic carrier by one declared O(3) action."""

        carriers = self._normalize_carriers(geometry_carriers)
        actions = {}
        transformed = {}
        for key, carrier in carriers.items():
            action_key = (int(key[1]), str(key[2]))
            if action_key not in actions:
                actions[action_key] = _real_tesseral_o3_action(
                    action_key[0],
                    action_key[1],
                    matrix,
                    dtype=carrier.dtype,
                    device=carrier.device,
                )
            transformed[key] = torch.einsum(
                "mn,...baqn->...baqm",
                actions[action_key],
                carrier,
            )
        return transformed

    def get_extra_state(self):
        return {
            "convention_hash": self.convention_hash,
            "coefficient_hash": self.coefficient_hash,
            "binding_type_ids": tuple(self.binding_type_ids),
            "geometry_channels": tuple(
                (key[0], int(key[1]), key[2], int(value))
                for key, value in sorted(self.geometry_channels.items())
            ),
            "compiler": dict(self.compiler_payload),
        }

    def set_extra_state(self, state):
        if str(state.get("convention_hash", "")) != self.convention_hash:
            raise ValueError("Serialized synthesis convention hash does not match the compiled plan.")
        if str(state.get("coefficient_hash", "")) != self.coefficient_hash:
            raise ValueError("Serialized synthesis coefficient hash does not match the compiled plan.")
        if tuple(state.get("binding_type_ids", ())) != self.binding_type_ids:
            raise ValueError("Serialized binding types do not match the readout plan.")
        expected_channels = tuple(
            (key[0], int(key[1]), key[2], int(value))
            for key, value in sorted(self.geometry_channels.items())
        )
        if tuple(state.get("geometry_channels", ())) != expected_channels:
            raise ValueError("Serialized geometry channels do not match the readout plan.")
        self.compiler_payload = dict(state.get("compiler", self.compiler_payload))

    def backend_report(self):
        return {
            "runtime": "YE3TBindingAttachedAlgebraicCurvatureReadout",
            "compiler_owner": "ye3t.couplings.compile_algebraic_curvature_synthesis",
            "convention_hash": self.convention_hash,
            "coefficient_hash": self.coefficient_hash,
            "fixed_output_partition": (2, 2),
            "binding_attached": True,
            "binding_type_count": int(len(self._runtime_types)),
            "execution_group_count": int(len(self._runtime_groups)),
            "binding_count": int(len(self.binding_type_ids)),
            "zero_image_binding_count": int(self._inactive_binding_count),
            "learned_axes": ("binding_type", "geometry_channel"),
            "complete_axes": (
                "physical_binding",
                "output_multiplicity",
                "magnetic",
                "tableau",
            ),
            "intertwiner_policy": "central_identity_on_output_multiplicity",
            "intertwiner_basis_normalization": (
                "compiler_Frobenius_normalization_absorbed_into_trainable_channel_weight"
            ),
            "global_atom_label_representation_materialized": False,
            "dense_global_basis_materialized": False,
            "dense_ambient_projector_materialized": False,
            "forward_backend": "torch_destination_grouped_template_then_sparse_compact_scatter",
            "adjoint": "torch_autograd_exact_linear_transpose",
            "double_adjoint": "torch_autograd_exact_linear_map",
            "complete_carrier_o3_transport": "YE3T_real_tesseral_signed_O3",
        }


__all__ = [
    "YE3TAlgebraicCurvatureOutput",
    "YE3TAlgebraicCurvatureReadout",
    "YE3TBindingAttachedAlgebraicCurvatureReadout",
]
