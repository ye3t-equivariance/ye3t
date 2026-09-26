
from collections.abc import Sequence as SequenceABC
from collections import defaultdict

from ye3t.core import YE3TAPI
from ye3t.core.irreps import IrrepTerm
from ye3t.core.product_descriptors import ExactProductColumnDescriptor
from .model import E3OperatorIR, ExactPath, PackedPathBlock, PathKey


def build_operator_ir(
    nin,
    lin,
    L_R,
    *,
    tree_type = 'balanced',
    factorization_policy = 'full',
    primitive_first = False,
    include_target_primitive = False,
    generator_ranks = None,
    generator_Ls = None,
    max_generator_rank = None,
    max_generator_L = None,
    max_recoupling_L = None,
    max_internal_L = None,
    name = None,
):
    if max_internal_L is not None and max_recoupling_L is None:
        max_recoupling_L = int(max_internal_L)
    api = YE3TAPI(tree_type=tree_type)
    exact_summary = api.summarize_ye3t(nin, lin)
    catalog = api.build_exact_path_catalog(
        nin,
        lin,
        L_R,
        factorization_policy=factorization_policy,
        include_target_primitive=include_target_primitive,
        primitive_first=primitive_first,
        generator_ranks=generator_ranks,
        generator_Ls=generator_Ls,
        max_generator_rank=max_generator_rank,
        max_generator_L=max_generator_L,
        max_recoupling_L=max_recoupling_L,
    )
    paths = tuple(_path_from_descriptor(d, primitive_first=primitive_first) for d in catalog.descriptors)
    packed = _pack_paths(paths)
    output_irreps = tuple(term for term in (exact_summary.irreps_out or tuple()) if int(term.l) == int(L_R))
    if not output_irreps:
        output_irreps = (IrrepTerm(catalog.target.dim, int(L_R), 'e'),) if int(catalog.target.dim) > 0 else tuple()
    return E3OperatorIR(
        name=name or f'ye3t_op_N{len(tuple(nin))}_L{int(L_R)}',
        input_irreps=tuple(exact_summary.irreps_out),
        output_irreps=tuple(output_irreps),
        target_nin=tuple(int(x) for x in nin),
        target_lin=tuple(int(x) for x in lin),
        target_L=int(L_R),
        primitive_first=bool(primitive_first),
        factorization_policy=str(factorization_policy),
        paths=paths,
        packed_blocks=packed,
        metadata={
            'tree_type': str(tree_type),
            'include_target_primitive': bool(include_target_primitive),
            'target_dim': int(catalog.target.dim),
            'primitive_rank': None if catalog.primitive is None else int(catalog.primitive.primitive_rank),
            'generated_rank': None if catalog.primitive is None else int(catalog.primitive.generated_rank),
            'raw_product_columns': None if catalog.subspace is None else int(catalog.subspace.raw_product_column_count),
            'independent_product_rank': None if catalog.subspace is None else int(catalog.subspace.independent_product_rank),
        },
    )


def resolve_target_Ls(
    nin,
    lin,
    L_R,
    *,
    tree_type = 'balanced',
    max_target_L = None,
):
    """Resolve a scalar, explicit list, or ``'all'`` target irrep request."""
    if isinstance(L_R, str):
        if L_R.lower() != 'all':
            raise ValueError("L_R as a string must be 'all'.")
        return YE3TAPI(tree_type=tree_type).allowed_target_Ls(nin, lin, max_target_L=max_target_L)
    if isinstance(L_R, SequenceABC) and not isinstance(L_R, (bytes, bytearray)):
        values = tuple(sorted({int(x) for x in L_R}))
    else:
        values = (int(L_R),)
    if max_target_L is not None:
        values = tuple(L for L in values if L <= int(max_target_L))
    return values


def build_operator_irs(
    nin,
    lin,
    L_R,
    *,
    tree_type = 'balanced',
    factorization_policy = 'full',
    primitive_first = False,
    include_target_primitive = False,
    generator_ranks = None,
    generator_Ls = None,
    max_generator_rank = None,
    max_generator_L = None,
    max_recoupling_L = None,
    max_internal_L = None,
    max_target_L = None,
    name = None,
):
    """Build one IR per requested target irrep.

    The expensive symbolic summary is shared through the underlying cached
    exact backend, while product catalogs remain target-sector specific because
    quotient/factorization ranks are target dependent.
    """
    targets = resolve_target_Ls(nin, lin, L_R, tree_type=tree_type, max_target_L=max_target_L)
    return tuple(
        build_operator_ir(
            nin,
            lin,
            int(target_L),
            tree_type=tree_type,
            factorization_policy=factorization_policy,
            primitive_first=primitive_first,
            include_target_primitive=include_target_primitive,
            generator_ranks=generator_ranks,
            generator_Ls=generator_Ls,
            max_generator_rank=max_generator_rank,
            max_generator_L=max_generator_L,
            max_recoupling_L=max_recoupling_L,
            max_internal_L=max_internal_L,
            name=None if name is None else f"{name}_L{int(target_L)}",
        )
        for target_L in targets
    )


def _path_from_descriptor(desc, *, primitive_first):
    key = _descriptor_to_key(desc)
    support = tuple(sorted(set(_descriptor_basis_indices(desc))))
    return ExactPath(key=key, descriptor=desc, support_indices=support, coefficient_count=len(support), primitive_first=bool(primitive_first))


def _descriptor_to_key(desc):
    if desc.kind == 'product' and desc.left is not None and desc.right is not None:
        return PathKey(int(desc.left.L_R), int(desc.right.L_R), int(desc.L_R))
    return PathKey(-1, -1, int(desc.L_R))


def _descriptor_basis_indices(desc):
    if desc.kind == 'primitive':
        return [] if desc.basis_index is None else [int(desc.basis_index)]
    out = []
    if desc.left is not None:
        out.extend(_descriptor_basis_indices(desc.left))
    if desc.right is not None:
        out.extend(_descriptor_basis_indices(desc.right))
    return out


def _pack_paths(paths):
    grouped = defaultdict(list)
    for path in paths:
        grouped[path.key].append(path)
    blocks = []
    for key, members in grouped.items():
        support_union = tuple(sorted({idx for member in members for idx in member.support_indices}))
        members_sorted = tuple(sorted(members, key=lambda p: (len(p.support_indices), p.coefficient_count)))
        blocks.append(PackedPathBlock(key=key, paths=members_sorted, support_union=support_union))
    return tuple(sorted(blocks, key=lambda b: (b.key.left_L, b.key.right_L, b.key.out_L, len(b.paths))))


__all__ = [
    'build_operator_ir',
    'build_operator_irs',
    'resolve_target_Ls',
    'E3OperatorIR',
    'ExactPath',
    'PackedPathBlock',
    'PathKey',
]
