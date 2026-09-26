
"""Optional OpenEquivariance lowering helpers.

The bridge is deliberately conservative: ye3t still owns exact algebra,
canonicalization, pruning, and primitive/product recipes.  OpenEquivariance is
used only as an optional backend for dense paired CG tensor-product blocks that
can be expressed as a weighted ``uvu`` problem.
"""

from dataclasses import field
from functools import lru_cache
import importlib
import importlib.util
import hashlib
import math

import torch
from ye3t._record import recordclass


@recordclass(('available', 'reason', 'version'), frozen = True)
class OpenEquivarianceAvailability:
    """Availability report for the optional OpenEquivariance backend."""
    version = None


@recordclass(('left_L', 'right_L', 'out_L', 'channel_count', 'irreps_in1', 'irreps_in2', 'irreps_out', 'instructions', 'connection_mode', 'layout', 'irrep_normalization', 'path_normalization', 'shared_weights', 'weight_policy', 'metadata'), frozen = True)
class OpenEquivariancePairedCGSpec:
    """One paired-channel ``uvu`` CG block that may be lowered to OEQ."""
    connection_mode = "uvu"
    layout = "mul_ir"
    irrep_normalization = "component"
    path_normalization = "none"
    shared_weights = True
    weight_policy = "shared_identity_diagonal"
    metadata = None

    def as_dict(self):
        return {
            "left_L": int(self.left_L),
            "right_L": int(self.right_L),
            "out_L": int(self.out_L),
            "channel_count": int(self.channel_count),
            "irreps_in1": str(self.irreps_in1),
            "irreps_in2": str(self.irreps_in2),
            "irreps_out": str(self.irreps_out),
            "instructions": [tuple(instruction) for instruction in self.instructions],
            "connection_mode": str(self.connection_mode),
            "layout": str(self.layout),
            "irrep_normalization": str(self.irrep_normalization),
            "path_normalization": str(self.path_normalization),
            "shared_weights": bool(self.shared_weights),
            "weight_policy": str(self.weight_policy),
            "metadata": dict(self.metadata or {}),
        }


@recordclass(('segment_index', 'left_L', 'right_L', 'out_L', 'path_count', 'channel_count', 'product_order', 'subtree_fingerprint', 'structured_subtree_signature', 'repeated_sector_signature', 'repeated_sector_metadata', 'supported', 'reason', 'spec'), frozen = True)
class OpenEquivarianceBlockPlan:
    """One exact schedule block classified for optional OEQ lowering."""
    spec = None

    @property
    def schedule_key(self):
        payload = "|".join(
            [
                str(int(self.left_L)),
                str(int(self.right_L)),
                str(int(self.out_L)),
                str(int(self.channel_count)),
                str(int(self.product_order)),
                str(self.subtree_fingerprint),
                str(self.structured_subtree_signature),
                str(self.repeated_sector_signature),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self):
        return {
            "segment_index": int(self.segment_index),
            "left_L": int(self.left_L),
            "right_L": int(self.right_L),
            "out_L": int(self.out_L),
            "path_count": int(self.path_count),
            "channel_count": int(self.channel_count),
            "product_order": int(self.product_order),
            "subtree_fingerprint": str(self.subtree_fingerprint),
            "structured_subtree_signature": str(self.structured_subtree_signature),
            "repeated_sector_signature": str(self.repeated_sector_signature),
            "repeated_sector_metadata": dict(self.repeated_sector_metadata),
            "schedule_key": self.schedule_key,
            "supported": bool(self.supported),
            "reason": str(self.reason),
            "spec": None if self.spec is None else self.spec.as_dict(),
        }


@recordclass(('operator_name', 'blocks', 'availability', 'metadata'), frozen = True)
class OpenEquivarianceLoweringPlan:
    """Optional OEQ backend plan after exact ye3t algebraic scheduling."""
    metadata = field(default_factory=dict)

    @property
    def candidate_block_count(self):
        return sum(1 for block in self.blocks if block.supported)

    @property
    def fallback_block_count(self):
        return sum(1 for block in self.blocks if not block.supported)

    @property
    def candidate_path_count(self):
        return sum(int(block.path_count) for block in self.blocks if block.supported)

    def as_dict(self):
        return {
            "operator_name": str(self.operator_name),
            "availability": self.availability.__dict__,
            "candidate_block_count": int(self.candidate_block_count),
            "fallback_block_count": int(self.fallback_block_count),
            "candidate_path_count": int(self.candidate_path_count),
            "blocks": [block.as_dict() for block in self.blocks],
            "metadata": dict(self.metadata),
            "fallback": "torch/triton/native_ye3t",
        }


def _descriptor_order(descriptor):
    if getattr(descriptor, "kind", "") == "primitive":
        return 1
    left = getattr(descriptor, "left", None)
    right = getattr(descriptor, "right", None)
    return _descriptor_order(left) + _descriptor_order(right) if left is not None and right is not None else 1


def _descriptor_subtree_payload(descriptor):
    kind = str(getattr(descriptor, "kind", "unknown"))
    L_R = int(getattr(descriptor, "L_R", -1))
    if kind == "primitive":
        label = getattr(descriptor, "basis_label", None)
        handle = getattr(descriptor, "basis_handle", None)
        if label is not None:
            leaf = repr(label)
        elif handle is not None:
            leaf = repr(handle)
        else:
            leaf = "anonymous"
        return f"primitive:L{L_R}:{leaf}"
    left = getattr(descriptor, "left", None)
    right = getattr(descriptor, "right", None)
    return f"product:L{L_R}:({_descriptor_subtree_payload(left)} x {_descriptor_subtree_payload(right)})"


def _descriptor_subtree_shape_payload(descriptor):
    """Compact tree-shape signature that preserves balanced subtree structure."""

    kind = str(getattr(descriptor, "kind", "unknown"))
    L_R = int(getattr(descriptor, "L_R", -1))
    if kind == "primitive":
        return f"primitive:L{L_R}"
    left = getattr(descriptor, "left", None)
    right = getattr(descriptor, "right", None)
    return f"product:L{L_R}:({_descriptor_subtree_shape_payload(left)} x {_descriptor_subtree_shape_payload(right)})"


def _subtree_fingerprint(descriptors):
    payload = "\n".join(_descriptor_subtree_payload(descriptor) for descriptor in descriptors)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _structured_subtree_signature(descriptors):
    payload = "\n".join(_descriptor_subtree_shape_payload(descriptor) for descriptor in descriptors)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _canonical_l_first_representative(nin, lin):
    grouped = {}
    for n_value, l_value in zip(tuple(nin), tuple(lin)):
        grouped.setdefault(int(l_value), []).append(int(n_value))
    ordered_blocks = sorted(grouped.items(), key=lambda item: (-len(item[1]), int(item[0])))
    canonical_n = []
    canonical_l = []
    repeated_l_blocks = []
    for l_value, n_values in ordered_blocks:
        sorted_n = sorted(int(n) for n in n_values)
        canonical_l.extend([int(l_value)] * len(sorted_n))
        canonical_n.extend(sorted_n)
        if len(sorted_n) > 1:
            repeated_l_blocks.append((int(l_value), len(sorted_n)))
    return {
        "canonical_nin": tuple(canonical_n),
        "canonical_lin": tuple(canonical_l),
        "repeated_l_blocks": tuple(repeated_l_blocks),
    }


def _repeated_sector_metadata(
    ir,
    *,
    left_L,
    right_L,
    out_L,
    product_order,
    support_width,
    structured_subtree_signature,
):
    canonical = _canonical_l_first_representative(
        tuple(int(x) for x in tuple(getattr(ir, "target_nin", ()))),
        tuple(int(x) for x in tuple(getattr(ir, "target_lin", ()))),
    )
    tree_type = str(getattr(ir, "metadata", {}).get("tree_type", "unknown"))
    payload = "|".join(
        [
            f"n={canonical['canonical_nin']}",
            f"l={canonical['canonical_lin']}",
            f"tree={tree_type}",
            f"target_L={int(getattr(ir, 'target_L', out_L))}",
            f"block=({int(left_L)},{int(right_L)},{int(out_L)})",
            f"order={int(product_order)}",
            f"support={int(support_width)}",
            f"subtree={structured_subtree_signature}",
        ]
    )
    return {
        **canonical,
        "tree_type": tree_type,
        "target_L": int(getattr(ir, "target_L", out_L)),
        "left_L": int(left_L),
        "right_L": int(right_L),
        "out_L": int(out_L),
        "product_order": int(product_order),
        "support_width": int(support_width),
        "structured_subtree_signature": str(structured_subtree_signature),
        "signature": hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16],
    }


def _block_is_paired_product(block):
    paths = tuple(getattr(block, "paths", ()))
    if not paths:
        return False, "empty block"
    shape_payloads = set()
    for path in paths:
        descriptor = getattr(path, "descriptor", None)
        if getattr(descriptor, "kind", "") != "product":
            return False, "contains primitive-only or non-product descriptor"
        left = getattr(descriptor, "left", None)
        right = getattr(descriptor, "right", None)
        if left is None or right is None:
            return False, "product descriptor is missing children"
        if int(getattr(left, "L_R", -999)) != int(getattr(block.key, "left_L", -1)):
            return False, "left child angular label does not match block"
        if int(getattr(right, "L_R", -999)) != int(getattr(block.key, "right_L", -1)):
            return False, "right child angular label does not match block"
        if int(getattr(descriptor, "L_R", -999)) != int(getattr(block.key, "out_L", -1)):
            return False, "output angular label does not match block"
        shape_payloads.add(_descriptor_subtree_shape_payload(descriptor))
    if len(shape_payloads) > 1:
        return False, "unsupported mixed subtree shapes"
    return True, "paired product block"


def _identity_diagonal_weight(channel_count, *, dtype, device):
    """Shared ``uvu`` weights that select ``left_channel == right_channel``.

    OpenEquivariance supports ``uvu`` but not the narrower ``uuu`` mode.  A
    paired ye3t block is therefore represented as ``uvu`` with an identity
    matrix over the left/right multiplicities.  If OEQ is available, this vector
    is reordered through ``TensorProduct.reorder_weights_from_e3nn`` before use.
    """

    eye = torch.eye(int(channel_count), dtype=dtype, device=device)
    return eye.reshape(-1).contiguous()


def _openequivariance_weight_for_problem(module, channel_count, *, dtype, device):
    weight = _identity_diagonal_weight(int(channel_count), dtype=dtype, device=device)
    reorder = getattr(module, "reorder_weights_from_e3nn", None)
    if callable(reorder):
        try:
            reordered = reorder(weight, has_batch_dim=False)
            if isinstance(reordered, torch.Tensor):
                weight = reordered.to(device=device, dtype=dtype).contiguous()
        except Exception:
            pass
    return weight


def _ye3t_real_cg_tensor_cpu(left_L, right_L, out_L):
    from ye3t.paired_cg import _real_cg_entries_cpu

    tensor = torch.zeros(
        (2 * int(left_L) + 1, 2 * int(right_L) + 1, 2 * int(out_L) + 1),
        dtype=torch.float64,
    )
    for index_left, index_right, index_out, value in _real_cg_entries_cpu(
        int(left_L), int(right_L), int(out_L)
    ):
        tensor[index_left, index_right, index_out] = float(value)
    return tensor


def _openequivariance_component_cg_tensor_cpu(left_L, right_L, out_L):
    module = importlib.import_module("openequivariance.core.e3nn_lite")
    coefficients = module.wigner_3j(
        int(left_L), int(right_L), int(out_L)
    )
    return math.sqrt(2 * int(out_L) + 1) * torch.as_tensor(
        coefficients,
        dtype=torch.float64,
    )


@lru_cache(maxsize=64)
def _openequivariance_basis_transform_cpu(angular_L):
    """Map YE3T real-tesseral rows into OpenEquivariance coordinates."""

    angular_L = int(angular_L)
    if angular_L < 0:
        raise ValueError("angular_L must be nonnegative")
    if angular_L == 0:
        return torch.ones((1, 1), dtype=torch.float64)
    if angular_L == 1:
        return torch.eye(3, dtype=torch.float64)
    left_transform = _openequivariance_basis_transform_cpu(angular_L - 1)
    right_transform = _openequivariance_basis_transform_cpu(1)
    oeq_tensor = _openequivariance_component_cg_tensor_cpu(
        angular_L - 1, 1, angular_L
    )
    target = _ye3t_real_cg_tensor_cpu(angular_L - 1, 1, angular_L)
    transformed_inputs = torch.einsum(
        "ai,bj,ijk->abk",
        left_transform,
        right_transform,
        oeq_tensor,
    )
    output_inverse = (
        transformed_inputs.reshape(-1, 2 * angular_L + 1).T
        @ target.reshape(-1, 2 * angular_L + 1)
    )
    transform = output_inverse.T
    identity = torch.eye(2 * angular_L + 1, dtype=torch.float64)
    orthogonality_residual = torch.max(
        torch.abs(transform.T @ transform - identity)
    ).item()
    reconstruction = torch.einsum(
        "abk,kc->abc",
        transformed_inputs,
        transform.T,
    )
    reconstruction_residual = torch.max(
        torch.abs(reconstruction - target)
    ).item()
    if orthogonality_residual > 5.0e-11 or reconstruction_residual > 5.0e-11:
        raise RuntimeError(
            "OpenEquivariance basis transform failed exact CG validation: "
            f"L={angular_L}, orthogonality={orthogonality_residual:.3e}, "
            f"reconstruction={reconstruction_residual:.3e}."
        )
    return transform.contiguous()


@lru_cache(maxsize=256)
def _openequivariance_cg_scale_cpu(left_L, right_L, out_L):
    left_transform = _openequivariance_basis_transform_cpu(int(left_L))
    right_transform = _openequivariance_basis_transform_cpu(int(right_L))
    output_transform = _openequivariance_basis_transform_cpu(int(out_L))
    candidate = torch.einsum(
        "ai,bj,ijk,kc->abc",
        left_transform,
        right_transform,
        _openequivariance_component_cg_tensor_cpu(left_L, right_L, out_L),
        output_transform.T,
    )
    target = _ye3t_real_cg_tensor_cpu(left_L, right_L, out_L)
    denominator = torch.sum(candidate * candidate)
    scale = torch.sum(candidate * target) / denominator
    residual = torch.max(torch.abs(scale * candidate - target)).item()
    if residual > 5.0e-11:
        raise RuntimeError(
            "OpenEquivariance CG convention failed YE3T validation: "
            f"({int(left_L)},{int(right_L)}->{int(out_L)}), "
            f"residual={residual:.3e}."
        )
    return float(scale.item())


def _openequivariance_bridge_tensors(
    left_L,
    right_L,
    out_L,
    *,
    dtype,
    device,
):
    left_transform = _openequivariance_basis_transform_cpu(int(left_L)).to(
        dtype=dtype,
        device=device,
    )
    right_transform = _openequivariance_basis_transform_cpu(int(right_L)).to(
        dtype=dtype,
        device=device,
    )
    output_inverse = _openequivariance_basis_transform_cpu(int(out_L)).T.to(
        dtype=dtype,
        device=device,
    )
    scale = _openequivariance_cg_scale_cpu(
        int(left_L), int(right_L), int(out_L)
    )
    return left_transform, right_transform, output_inverse, scale


def _parity_for_product(left_L, right_L, out_L):
    left_parity = "e" if int(left_L) % 2 == 0 else "o"
    right_parity = "e" if int(right_L) % 2 == 0 else "o"
    out_even = (int(left_L) + int(right_L)) % 2 == 0
    out_parity = "e" if out_even else "o"
    return left_parity, right_parity, out_parity


def _irrep_string(channel_count, L, parity):
    return f"{int(channel_count)}x{int(L)}{parity}"


def _spec_to_problem_kwargs(spec):
    return {
        "irreps_in1": spec.irreps_in1,
        "irreps_in2": spec.irreps_in2,
        "irreps_out": spec.irreps_out,
        "instructions": list(spec.instructions),
        "internal_weights": False,
        "shared_weights": bool(spec.shared_weights),
        "irrep_normalization": spec.irrep_normalization,
        "path_normalization": spec.path_normalization,
        "layout": spec.layout,
    }


def openequivariance_availability():
    """Return whether the Python package is importable in this environment."""

    spec = importlib.util.find_spec("openequivariance")
    if spec is None:
        return OpenEquivarianceAvailability(
            available=False,
            reason="Python package 'openequivariance' is not installed or is not compatible with this interpreter.",
        )
    try:
        module = importlib.import_module("openequivariance")
    except Exception as exc:  # pragma: no cover - depends on optional package/GPU stack
        return OpenEquivarianceAvailability(available=False, reason=f"import failed: {exc}")
    version = getattr(module, "__version__", None)
    return OpenEquivarianceAvailability(available=True, reason="available", version=None if version is None else str(version))


def make_paired_uvu_spec(
    left_L,
    right_L,
    out_L,
    channel_count,
    *,
    parity = None,
):
    """Build an OEQ-style ``uvu`` spec for paired product channels.

    ye3t's native packed path has shape ``[batch, channel, 2L+1]`` for both
    children and produces the same channel index at the output.  That maps to a
    single e3nn/OpenEquivariance instruction over multiplicity-``channel``
    irreps.
    """

    channel_count = int(channel_count)
    if channel_count < 1:
        raise ValueError("channel_count must be positive.")
    left_L = int(left_L)
    right_L = int(right_L)
    out_L = int(out_L)
    left_parity, right_parity, out_parity = (
        (str(parity), str(parity), str(parity))
        if parity is not None
        else _parity_for_product(left_L, right_L, out_L)
    )
    return OpenEquivariancePairedCGSpec(
        left_L=left_L,
        right_L=right_L,
        out_L=out_L,
        channel_count=channel_count,
        irreps_in1=_irrep_string(channel_count, left_L, left_parity),
        irreps_in2=_irrep_string(channel_count, right_L, right_parity),
        irreps_out=_irrep_string(channel_count, out_L, out_parity),
        instructions=((0, 0, 0, "uvu", True, 1.0),),
        metadata={
            "bridge_kind": "paired_uvu_cg",
            "safe_fallback_required": True,
            "supported_by_openequivariance_docs": "pure weighted uvu tensor product",
            "source": "ye3t native packed paired CG segment",
        },
    )


def _oeq_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        import numpy as np

        return np.float32
    if torch_dtype == torch.float64:
        import numpy as np

        return np.float64
    raise TypeError(f"OpenEquivariance bridge only supports float32/float64, got {torch_dtype}.")


@lru_cache(maxsize=128)
def _build_cached_tensor_product(
    left_L,
    right_L,
    out_L,
    channel_count,
    dtype_name,
):
    """Build and cache an OEQ TensorProduct for one static paired block.

    OpenEquivariance JIT compilation can take seconds, so construction is
    cached by static angular/multiplicity/dtype signature.  Initialization can
    still fail when no visible GPU exists or the problem is unsupported; callers
    catch that and fall back.
    """

    oeq = importlib.import_module("openequivariance")
    dtype = torch.float32 if dtype_name == "torch.float32" else torch.float64
    np_dtype = _oeq_dtype(dtype)
    spec = make_paired_uvu_spec(int(left_L), int(right_L), int(out_L), int(channel_count))
    problem = oeq.TPProblem(
        oeq.Irreps(spec.irreps_in1),
        oeq.Irreps(spec.irreps_in2),
        oeq.Irreps(spec.irreps_out),
        list(spec.instructions),
        internal_weights=False,
        shared_weights=True,
        irrep_normalization=spec.irrep_normalization,
        path_normalization=spec.path_normalization,
        irrep_dtype=np_dtype,
        weight_dtype=np_dtype,
        layout=spec.layout,
        label=f"ye3t_L{left_L}_x_L{right_L}_to_L{out_L}_C{channel_count}",
    )
    module = oeq.TensorProduct(problem)
    return module, problem, spec


def openequivariance_paired_cg_forward(
    left,
    right,
    *,
    left_L,
    right_L,
    out_L,
):
    """Try an OpenEquivariance paired ``uvu`` product, returning ``None`` on fallback.

    This function never raises for normal unsupported-backend conditions.  The
    native ye3t runtime can therefore use it in a best-effort path before
    falling back to Triton or Torch.
    """

    if left.ndim != 3 or right.ndim != 3 or left.shape[:2] != right.shape[:2]:
        return None
    if left.dtype not in (torch.float32, torch.float64) or right.dtype != left.dtype:
        return None
    if not left.is_cuda or not right.is_cuda:
        return None
    availability = openequivariance_availability()
    if not availability.available:
        return None
    try:
        module, problem, _spec = _build_cached_tensor_product(
            int(left_L),
            int(right_L),
            int(out_L),
            int(left.shape[1]),
            str(left.dtype),
        )
        if hasattr(module, "to"):
            module = module.to(device=left.device)
        (
            left_transform,
            right_transform,
            output_inverse,
            cg_scale,
        ) = _openequivariance_bridge_tensors(
            int(left_L),
            int(right_L),
            int(out_L),
            dtype=left.dtype,
            device=left.device,
        )
        x = torch.matmul(left, left_transform).contiguous().reshape(
            int(left.shape[0]), -1
        )
        y = torch.matmul(right, right_transform).contiguous().reshape(
            int(right.shape[0]), -1
        )
        weight = _openequivariance_weight_for_problem(module, int(left.shape[1]), dtype=left.dtype, device=left.device)
        weight = float(cg_scale) * weight
        if int(weight.numel()) != int(problem.weight_numel):
            return None
        out_flat = module(x, y, weight)
        out = torch.matmul(
            out_flat.reshape(
                int(left.shape[0]),
                int(left.shape[1]),
                2 * int(out_L) + 1,
            ),
            output_inverse,
        )
        return out.contiguous(), "openequivariance_paired_uvu"
    except Exception:
        return None


def build_openequivariance_lowering_plan(ir_or_schedule):
    """Classify exact schedule blocks for optional OEQ lowering.

    The plan deliberately preserves ye3t's algebraic metadata.  Subtree
    fingerprints let us identify repeated exact-product structure before any
    backend sees a dense CG block, so future planners can reuse subtrees and
    send only the reduced dense blocks to OpenEquivariance.
    """

    schedule = ir_or_schedule
    ir = getattr(schedule, "operator", ir_or_schedule)
    segments = tuple(getattr(schedule, "segments", ()))
    blocks = tuple(getattr(schedule, "retained_blocks", getattr(ir, "packed_blocks", ())))
    plans = []
    descriptor_fingerprints = []
    for segment_index, segment in enumerate(segments):
        block = blocks[segment_index] if segment_index < len(blocks) else None
        left_L = int(getattr(segment, "left_L", -1))
        right_L = int(getattr(segment, "right_L", -1))
        out_L = int(getattr(segment, "out_L", -1))
        path_count = int(getattr(segment, "num_paths", 0))
        channel_count = len(tuple(getattr(segment, "packed_support", ()))) or path_count
        descriptors = tuple(getattr(path, "descriptor", None) for path in tuple(getattr(block, "paths", ()))) if block is not None else tuple()
        product_order = max((_descriptor_order(descriptor) for descriptor in descriptors), default=0)
        subtree_fingerprint = _subtree_fingerprint(descriptors)
        structured_subtree_signature = _structured_subtree_signature(descriptors)
        descriptor_fingerprints.extend(_subtree_fingerprint((descriptor,)) for descriptor in descriptors)
        repeated_sector_metadata = _repeated_sector_metadata(
            ir,
            left_L=left_L,
            right_L=right_L,
            out_L=out_L,
            product_order=product_order,
            support_width=len(tuple(getattr(segment, "packed_support", ()))),
            structured_subtree_signature=structured_subtree_signature,
        )
        supported = False
        reason = "unsupported segment"
        spec = None
        if block is None:
            reason = "missing retained block"
        elif left_L < 0 or right_L < 0:
            reason = "primitive segment stays in ye3t"
        elif channel_count <= 0:
            reason = "empty channel set"
        else:
            paired, paired_reason = _block_is_paired_product(block)
            if paired:
                supported = True
                reason = "pure weighted uvu candidate"
                spec = make_paired_uvu_spec(left_L, right_L, out_L, int(channel_count))
            else:
                reason = paired_reason
        plans.append(
            OpenEquivarianceBlockPlan(
                segment_index=int(segment_index),
                left_L=left_L,
                right_L=right_L,
                out_L=out_L,
                path_count=path_count,
                channel_count=int(channel_count),
                product_order=int(product_order),
                subtree_fingerprint=subtree_fingerprint,
                structured_subtree_signature=structured_subtree_signature,
                repeated_sector_signature=str(repeated_sector_metadata["signature"]),
                repeated_sector_metadata=repeated_sector_metadata,
                supported=bool(supported),
                reason=reason,
                spec=spec,
            )
        )
    raw_subproblem_count = len(descriptor_fingerprints)
    unique_subproblem_count = len(set(descriptor_fingerprints))
    reuse_ratio = (
        float(raw_subproblem_count) / float(unique_subproblem_count)
        if unique_subproblem_count
        else 0.0
    )
    reuse_fraction = (
        1.0 - (float(unique_subproblem_count) / float(raw_subproblem_count))
        if raw_subproblem_count
        else 0.0
    )
    candidate_block_count = sum(1 for block in plans if block.supported)
    fallback_block_count = sum(1 for block in plans if not block.supported)
    candidate_path_count = sum(int(block.path_count) for block in plans if block.supported)
    return OpenEquivarianceLoweringPlan(
        operator_name=str(getattr(ir, "name", "unknown")),
        blocks=tuple(plans),
        availability=openequivariance_availability(),
        metadata={
            "kernel_strategy": "ye3t_algebraic_schedule_then_openequivariance_weighted_uvu",
            "supported_connection_modes": ("uvu", "uvw"),
            "implemented_connection_modes": ("uvu",),
            "all_instructions_weighted": True,
            "internal_weights": False,
            "layout": "mul_ir",
            "subtree_fingerprints_preserved": True,
            "canonical_l_first_representative": _canonical_l_first_representative(
                tuple(int(x) for x in tuple(getattr(ir, "target_nin", ()))),
                tuple(int(x) for x in tuple(getattr(ir, "target_lin", ()))),
            ),
            "tree_type": str(getattr(ir, "metadata", {}).get("tree_type", "unknown")),
            "raw_symbolic_subproblem_count": int(raw_subproblem_count),
            "unique_subtree_fingerprint_count": int(unique_subproblem_count),
            "subtree_reuse_ratio": float(reuse_ratio),
            "subtree_reuse_fraction": float(reuse_fraction),
            "candidate_block_count": int(candidate_block_count),
            "fallback_block_count": int(fallback_block_count),
            "candidate_path_count": int(candidate_path_count),
            "diagnostics_are_compact": True,
        },
    )


def export_to_openequivariance_specs(ir_or_schedule):
    """Return a backend-agnostic export summary for OEQ-capable blocks.

    This is intentionally a report/export helper rather than a full compiler:
    exact ye3t schedules may contain primitive segments, decomposable recipes,
    and unsupported mixed products that should remain under ye3t control.
    """

    plan = build_openequivariance_lowering_plan(ir_or_schedule)
    payload = plan.as_dict()
    payload["candidate_blocks"] = [
        block["spec"] | {
            "segment_index": block["segment_index"],
            "path_count": block["path_count"],
            "product_order": block["product_order"],
            "subtree_fingerprint": block["subtree_fingerprint"],
            "supported_mode": "uvu",
        }
        for block in payload["blocks"]
        if block["supported"] and block["spec"] is not None
    ]
    return payload


__all__ = [
    "OpenEquivarianceAvailability",
    "OpenEquivarianceBlockPlan",
    "OpenEquivarianceLoweringPlan",
    "OpenEquivariancePairedCGSpec",
    "build_openequivariance_lowering_plan",
    "export_to_openequivariance_specs",
    "make_paired_uvu_spec",
    "openequivariance_availability",
    "openequivariance_paired_cg_forward",
]
