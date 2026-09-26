"""Count valid YE3T coupling labels for a fixed-content ACE sector."""

# coupling_multiplicity_counts.py
# Use ye3t.couplings.count as the single source of truth for valid labels and
# multiplicities. Edit content/input_Ls/target_L to inspect another sector.

from ye3t import couplings
from ye3t.workflows import merge_workflow_config


# YE3T config dictionary.
cfg_ye3t = {
    "content": (1, 1, 2, 2),  # Fixed non-angular channel content.
    "input_Ls": (1, 1, 2, 2),  # Slot angular momenta for the same content.
    "target_L": 0,  # Target angular momentum L_R.
    "target_permutation": "trivial",  # ACE invariant sector.
    "carrier": "ACE_density",  # Ordinary compact ACE density carrier.
    "label_preview": 2,  # Number of valid labels to print.
}


def run_coupling_multiplicity_counts(config=None):
    """Print a compact multiplicity report from ye3t.couplings.count."""
    settings = merge_workflow_config(cfg_ye3t, config)
    report = couplings.count(
        content=tuple(settings["content"]),
        input_Ls=tuple(settings["input_Ls"]),
        target_L=int(settings["target_L"]),
        target_permutation=str(settings["target_permutation"]),
        carrier=str(settings["carrier"]),
    )
    counts_by_target = dict(report.counts_by_target)
    labels = list(report.labels_for_target(int(settings["target_L"])))
    print("coupling count source:", "ye3t.couplings.count")
    print("content:", tuple(report.content))
    print("carrier:", report.carrier)
    print("target:", report.target)
    print("counts by target:", counts_by_target)
    print("total labels for target:", len(labels))
    print("validation passed:", bool(report.validation_report.get("passed", False)))
    for index, label in enumerate(labels[: max(0, int(settings["label_preview"]))]):
        print("label", index, label)


if __name__ == "__main__":
    run_coupling_multiplicity_counts(config=cfg_ye3t)
