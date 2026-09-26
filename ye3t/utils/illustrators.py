"""Illustration helpers for YE3T workflow examples."""

from pathlib import Path
import time
from xml.sax.saxutils import escape

import torch

from ye3t import AngularIrrep, CoupledIrrepLabel, Partition, PermutationIrrep, PermutationSubgroup
from ye3t.representations import (
    ExactSymbolicProjectorGeneralizedBasisBuilder,
    build_exact_lowered_change_of_group_basis_map,
    format_lowered_change_of_group_basis_map_report,
)
from ye3t.runtime.generalized import (
    GeneralizedExactRuntimeBlock,
    GeneralizedExactRuntimeIrreps,
    GeneralizedFullTensorProduct,
    GeneralizedIrreps,
    GeneralizedLinear,
)
from ye3t.runtime.symmetric_power import (
    allowed_symmetric_power_outputs,
    allowed_symmetric_square_outputs,
    symmetric_power_real_tesseral,
    symmetric_power_reference_real_tesseral,
    symmetric_square_real_tesseral,
)


def _as_output_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _svg_label(text):
    return escape(str(text), {'"': "&quot;"})


def _svg_text(x, y, text, size=12, weight="400", fill="#1f2933", anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Inter, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{_svg_label(text)}</text>'
    )


def _svg_node(x, y, width, height, title, lines=(), fill="#f8fafc", stroke="#334e68"):
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>',
        _svg_text(x + width / 2, y + 23, title, size=13, weight="700", fill="#102a43"),
    ]
    for index, line in enumerate(lines):
        parts.append(_svg_text(x + width / 2, y + 44 + 17 * index, line, size=11, fill="#334e68"))
    return "\n".join(parts)


def _svg_arrow(x1, y1, x2, y2, label=None, stroke="#486581", dash=False):
    dash_attr = ' stroke-dasharray="5 4"' if dash else ""
    parts = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="1.8" marker-end="url(#arrow)"{dash_attr}/>'
    ]
    if label:
        parts.append(_svg_text((x1 + x2) / 2, (y1 + y2) / 2 - 6, label, size=10, fill=stroke))
    return "\n".join(parts)


def _svg_shell(width, height, title, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_svg_label(title)}">
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#486581"/>
  </marker>
</defs>
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
{_svg_text(width / 2, 30, title, size=19, weight="700", fill="#102a43")}
{body}
</svg>
'''


def tensor_product_instruction_graph_svg(product, title=None, max_instructions=10):
    """Render the actual instruction graph from a constructed tensor product."""
    report = product.instruction_report()
    instructions = list(report["instructions"])[: int(max_instructions)]
    hidden_count = max(0, int(report["num_instructions"]) - len(instructions))
    row_height = 76
    width = 1080
    height = 108 + max(1, len(instructions)) * row_height + (32 if hidden_count else 0)
    title = title or "Constructed YE3T tensor-product instruction graph"
    body = [
        _svg_text(86, 72, "left block", size=13, weight="700", fill="#334e68"),
        _svg_text(270, 72, "right block", size=13, weight="700", fill="#334e68"),
        _svg_text(512, 72, "actual instruction", size=13, weight="700", fill="#334e68"),
        _svg_text(790, 72, "output sector", size=13, weight="700", fill="#334e68"),
    ]
    left_blocks = {int(row["index"]): row for row in report["irreps_in1"]["blocks"]}
    right_blocks = {int(row["index"]): row for row in report["irreps_in2"]["blocks"]}
    for index, instruction in enumerate(instructions):
        y = 100 + index * row_height
        left = left_blocks[int(instruction["left_block_index"])]
        right = right_blocks[int(instruction["right_block_index"])]
        output = instruction["output_label"]
        body.append(
            _svg_node(
                28,
                y,
                118,
                50,
                f"L={left['L']}",
                (f"block {left['index']}", f"dim {left['block_dim']}"),
                fill="#e0f2fe",
                stroke="#0f609b",
            )
        )
        body.append(
            _svg_node(
                212,
                y,
                118,
                50,
                f"L={right['L']}",
                (f"block {right['index']}", f"dim {right['block_dim']}"),
                fill="#ecfccb",
                stroke="#3f6212",
            )
        )
        body.append(
            _svg_node(
                418,
                y - 9,
                214,
                68,
                f"instr {instruction['index']}: L={instruction['output_L']}",
                (
                    f"source nnz {instruction['source_tensor_nnz']}",
                    f"coord nnz {instruction['coordinate_nnz']}",
                    f"young mult {instruction['young_multiplicity']}",
                ),
                fill="#fef3c7",
                stroke="#92400e",
            )
        )
        body.append(
            _svg_node(
                724,
                y,
                190,
                50,
                output,
                (f"out block {instruction['output_block_index']}",),
                fill="#ede9fe",
                stroke="#5b21b6",
            )
        )
        body.append(_svg_arrow(146, y + 25, 418, y + 25, "left"))
        body.append(_svg_arrow(330, y + 25, 418, y + 25, "right"))
        body.append(_svg_arrow(632, y + 25, 724, y + 25, "CG x Young"))
        body.append(
            _svg_text(
                944,
                y + 18,
                f"src {tuple(instruction['source_tensor_shape'])}",
                size=10,
                fill="#52606d",
                anchor="start",
            )
        )
        body.append(
            _svg_text(
                944,
                y + 36,
                f"coord {tuple(instruction['coordinate_shape'])}",
                size=10,
                fill="#52606d",
                anchor="start",
            )
        )
    if hidden_count:
        body.append(
            _svg_text(
                width / 2,
                height - 22,
                f"... {hidden_count} additional instructions omitted; raise max_diagram_instructions to show more.",
                size=12,
                fill="#52606d",
            )
        )
    return _svg_shell(width, height, title, "\n".join(body))


def write_tensor_product_instruction_graph(product, path, title=None, max_instructions=10):
    """Write the actual instruction graph from a constructed tensor product."""
    path = _as_output_path(path)
    path.write_text(
        tensor_product_instruction_graph_svg(product, title=title, max_instructions=max_instructions),
        encoding="utf-8",
    )
    return path


def exact_schedule_segment_graph_svg(schedule, title=None, max_segments=12):
    """Render the actual lowered-segment graph from a constructed schedule."""
    segments = list(schedule.segments)[: int(max_segments)]
    hidden_count = max(0, len(schedule.segments) - len(segments))
    decisions = {}
    if getattr(schedule, "materialization_plan", None) is not None:
        decisions = {
            int(decision.segment_index): decision
            for decision in tuple(schedule.materialization_plan.decisions)
        }
    row_height = 72
    width = 1060
    height = 108 + max(1, len(segments)) * row_height + (32 if hidden_count else 0)
    title = title or "Constructed YE3T exact-schedule segment graph"
    body = [
        _svg_text(88, 72, "left L", size=13, weight="700", fill="#334e68"),
        _svg_text(248, 72, "right L", size=13, weight="700", fill="#334e68"),
        _svg_text(456, 72, "lowered segment", size=13, weight="700", fill="#334e68"),
        _svg_text(690, 72, "output L", size=13, weight="700", fill="#334e68"),
        _svg_text(884, 72, "materialization", size=13, weight="700", fill="#334e68"),
    ]
    for index, segment in enumerate(segments):
        y = 100 + index * row_height
        decision = decisions.get(index)
        action = "unspecified" if decision is None else str(decision.action)
        reuse = "n/a" if decision is None else str(int(decision.reuse_count))
        support = len(tuple(segment.packed_support))
        primitive = int(segment.left_L) < 0 or int(segment.right_L) < 0
        body.append(_svg_node(42, y, 88, 48, f"L={segment.left_L}", (), fill="#e0f2fe", stroke="#0f609b"))
        body.append(_svg_node(202, y, 88, 48, f"L={segment.right_L}", (), fill="#ecfccb", stroke="#3f6212"))
        body.append(
            _svg_node(
                374,
                y - 8,
                170,
                66,
                f"segment {index}",
                (
                    f"paths {segment.num_paths}",
                    f"support {support}",
                    "primitive" if primitive else "full",
                ),
                fill="#fef3c7",
                stroke="#92400e",
            )
        )
        body.append(_svg_node(648, y, 88, 48, f"L={segment.out_L}", (), fill="#ede9fe", stroke="#5b21b6"))
        body.append(
            _svg_node(
                820,
                y - 2,
                142,
                52,
                action,
                (f"reuse {reuse}",),
                fill="#f8fafc",
                stroke="#52606d",
            )
        )
        body.append(_svg_arrow(130, y + 24, 374, y + 24, "left"))
        body.append(_svg_arrow(290, y + 24, 374, y + 24, "right"))
        body.append(_svg_arrow(544, y + 24, 648, y + 24, "lower"))
        body.append(_svg_arrow(736, y + 24, 820, y + 24, "plan", dash=True))
    if hidden_count:
        body.append(
            _svg_text(
                width / 2,
                height - 22,
                f"... {hidden_count} additional segments omitted; raise max_diagram_segments to show more.",
                size=12,
                fill="#52606d",
            )
        )
    return _svg_shell(width, height, title, "\n".join(body))


def write_exact_schedule_segment_graph(schedule, path, title=None, max_segments=12):
    """Write the actual lowered-segment graph from a constructed schedule."""
    path = _as_output_path(path)
    path.write_text(
        exact_schedule_segment_graph_svg(schedule, title=title, max_segments=max_segments),
        encoding="utf-8",
    )
    return path


def illustrate_generalized_permutation_characters(settings):
    """Print characters and run a small generalized linear map."""
    subgroup = PermutationSubgroup.from_nl(tuple(settings["nin"]), tuple(settings["lin"]))
    symmetric_vector = CoupledIrrepLabel(
        angular=AngularIrrep(int(settings["symmetric_L"])),
        permutation=PermutationIrrep.trivial_for_subgroup(subgroup),
    )
    antisymmetric_scalar = CoupledIrrepLabel(
        angular=AngularIrrep(int(settings["antisymmetric_L"])),
        permutation=PermutationIrrep(subgroup=subgroup, partitions=(Partition(tuple(settings["antisymmetric_partition"])),)),
    )
    irreps = GeneralizedIrreps([(1, symmetric_vector), (1, antisymmetric_scalar)])

    print("Generalized irreps:", irreps.to_string())
    print("dimension:", irreps.dim)

    swap = [[tuple(settings["swap"])] for _ in tuple(settings["nin"])]
    print("permutation character for block swap:", irreps.permutation_character(swap))
    print(
        "joint character at a small z-rotation:",
        irreps.joint_character(alpha=float(settings["alpha"]), beta=0.0, gamma=0.0, permutations=swap),
    )

    layer = GeneralizedLinear(irreps, irreps, dtype=torch.float64)
    x = irreps.randn(int(settings["batch"]), -1, dtype=torch.float64)
    y = layer(x)
    print("GeneralizedLinear shape:", tuple(x.shape), "->", tuple(y.shape))


def illustrate_generalized_tensor_product_change_of_group(settings):
    """Print change-of-group metadata and evaluate one runtime tensor product."""
    trivial_s1 = PermutationIrrep.trivial_for_subgroup(PermutationSubgroup.from_nl((1,), (1,)))
    left_sector = ExactSymbolicProjectorGeneralizedBasisBuilder((1,), (1,), trivial_s1).build()
    right_sector = ExactSymbolicProjectorGeneralizedBasisBuilder((1,), (1,), trivial_s1).build()

    lowered_map = build_exact_lowered_change_of_group_basis_map(
        left_sector,
        right_sector,
        left_L=int(settings["left_L"]),
        right_L=int(settings["right_L"]),
        output_L=int(settings["output_L"]),
        tree_type=str(settings["tree_type"]),
        node_span=tuple(settings["node_span"]),
    )
    print(format_lowered_change_of_group_basis_map_report(lowered_map))
    print()

    runtime_left = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, left_sector, int(settings["left_L"]), 0),))
    runtime_right = GeneralizedExactRuntimeIrreps((GeneralizedExactRuntimeBlock(1, right_sector, int(settings["right_L"]), 0),))
    full_tp = GeneralizedFullTensorProduct(runtime_left, runtime_right, tree_type=str(settings["tree_type"]))

    print(runtime_left.format_report())
    print()
    print(full_tp.format_instruction_report())
    print()

    torch.manual_seed(int(settings["seed"]))
    x_left = torch.randn(int(settings["batch"]), runtime_left.dim, dtype=torch.float64)
    x_right = torch.randn(int(settings["batch"]), runtime_right.dim, dtype=torch.float64)
    out = full_tp(x_left, x_right)
    print("runtime tensor-product shapes:")
    print(" ", tuple(x_left.shape), "x", tuple(x_right.shape), "->", tuple(out.shape))
    if settings.get("write_diagrams", False):
        output_dir = Path(settings["output_dir"])
        path = write_tensor_product_instruction_graph(
            full_tp,
            output_dir / "pedagogical_joint_young_cg_product.svg",
            title="Constructed generalized tensor-product instructions",
            max_instructions=int(settings.get("max_diagram_instructions", 10)),
        )
        print("diagram output:", path)


def _elapsed(fn, runs):
    start = time.perf_counter()
    for _ in range(int(runs)):
        fn()
    return time.perf_counter() - start


def print_symmetric_power_output_catalog(cases):
    """Print a compact catalog of tested input and output angular momenta."""
    for power, input_Ls in cases:
        for L in input_Ls:
            if int(power) == 2:
                outputs = allowed_symmetric_square_outputs(int(L))
            else:
                outputs = allowed_symmetric_power_outputs(int(power), int(L))
            print(f"catalog power={power} l_in={L} outputs:", outputs)


def compare_symmetric_square(case):
    """Run one bounded symmetric-square correctness and timing comparison."""
    L = int(case["L"])
    output_L = int(case["output_L"])
    torch.manual_seed(int(case["seed"]))
    x = torch.randn(int(case["batch"]), 2 * L + 1, dtype=torch.float64)

    direct = symmetric_square_real_tesseral(x, L, output_L, optimization_policy="auto")
    regular = symmetric_square_real_tesseral(x, L, output_L, optimization_policy="off")
    max_difference = float((direct - regular).abs().max())

    for _ in range(int(case["warmups"])):
        symmetric_square_real_tesseral(x, L, output_L, optimization_policy="auto")
        symmetric_square_real_tesseral(x, L, output_L, optimization_policy="off")

    direct_s = _elapsed(lambda: symmetric_square_real_tesseral(x, L, output_L, optimization_policy="auto"), case["runs"])
    regular_s = _elapsed(lambda: symmetric_square_real_tesseral(x, L, output_L, optimization_policy="off"), case["runs"])

    print("power=2 outputs:", allowed_symmetric_square_outputs(L))
    print("power=2 max difference:", max_difference)
    print("power=2 direct seconds:", f"{direct_s:.6f}")
    print("power=2 reference seconds:", f"{regular_s:.6f}")
    print("power=2 speed ratio:", f"{regular_s / max(direct_s, 1.0e-12):.3f}")


def compare_symmetric_power(case):
    """Run one bounded higher-rank symmetric-power comparison."""
    power = int(case["power"])
    L = int(case["L"])
    output_L = int(case["output_L"])
    torch.manual_seed(int(case["seed"]))
    x = torch.randn(int(case["batch"]), 2 * L + 1, dtype=torch.float64)

    direct = symmetric_power_real_tesseral(x, power, L, output_L, optimization_policy="auto")
    regular = symmetric_power_reference_real_tesseral(x, power, L, output_L)
    max_difference = float((direct - regular).abs().max())

    for _ in range(int(case["warmups"])):
        symmetric_power_real_tesseral(x, power, L, output_L, optimization_policy="auto")
        symmetric_power_reference_real_tesseral(x, power, L, output_L)

    direct_s = _elapsed(lambda: symmetric_power_real_tesseral(x, power, L, output_L, optimization_policy="auto"), case["runs"])
    regular_s = _elapsed(lambda: symmetric_power_reference_real_tesseral(x, power, L, output_L), case["runs"])

    print(f"power={power} outputs:", allowed_symmetric_power_outputs(power, L))
    print(f"power={power} max difference:", max_difference)
    print(f"power={power} direct seconds:", f"{direct_s:.6f}")
    print(f"power={power} reference seconds:", f"{regular_s:.6f}")
    print(f"power={power} speed ratio:", f"{regular_s / max(direct_s, 1.0e-12):.3f}")
