#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def bootstrap_interval(values: np.ndarray, samples: int = 20000) -> list[float]:
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(0)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, np.float64)
    return {
        "mean": float(values.mean()),
        "subject_bootstrap_95ci": bootstrap_interval(values),
        "per_subject": values.tolist(),
        "positive_subjects": int((values > 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/runs/animation_multisequence"))
    parser.add_argument("--mouth-root", type=Path, default=Path("artifacts/runs/animation_multisubject_exp5"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    subjects = ("393", "404", "461", "477", "486")
    sequence_paths = {
        "head": args.root / "EXP_1_head",
        "eyes": args.root / "EXP_2_eyes",
        "mouth": args.mouth_root,
        "jaw": args.root / "EXP_8_jaw_1",
    }
    scales = ("8", "all")
    methods = ("hard", "adaptive", "free")
    regions = ("head", "face", "oral", "eyes")
    reports = {
        sequence: {
            scale: {
                subject: {
                    method: json.loads((root / f"n{scale}" / subject / method / "metrics.json").read_text())
                    for method in methods
                }
                for subject in subjects
            }
            for scale in scales
        }
        for sequence, root in sequence_paths.items()
    }

    comparisons = {}
    for sequence in sequence_paths:
        comparisons[sequence] = {}
        for scale in scales:
            comparisons[sequence][scale] = {}
            for region in regions:
                hard = np.asarray(
                    [reports[sequence][scale][subject]["hard"]["evaluation"][region]["heldout"]["masked_psnr"] for subject in subjects]
                )
                comparisons[sequence][scale][region] = {}
                for method in ("adaptive", "free"):
                    candidate = np.asarray(
                        [reports[sequence][scale][subject][method]["evaluation"][region]["heldout"]["masked_psnr"] for subject in subjects]
                    )
                    comparisons[sequence][scale][region][f"{method}_minus_hard"] = summary(candidate - hard)

    target_control = {
        "head": ("head", "oral"),
        "eyes": ("eyes", "oral"),
        "mouth": ("oral", "eyes"),
        "jaw": ("oral", "eyes"),
    }
    localization = {}
    for sequence, (target, control) in target_control.items():
        localization[sequence] = {}
        for scale in scales:
            values = []
            for subject in subjects:
                report = reports[sequence][scale][subject]
                target_effect = (
                    report["adaptive"]["evaluation"][target]["heldout"]["masked_psnr"]
                    - report["hard"]["evaluation"][target]["heldout"]["masked_psnr"]
                )
                control_effect = (
                    report["adaptive"]["evaluation"][control]["heldout"]["masked_psnr"]
                    - report["hard"]["evaluation"][control]["heldout"]["masked_psnr"]
                )
                values.append(target_effect - control_effect)
            localization[sequence][scale] = {
                "target_region": target,
                "control_region": control,
                "difference_of_differences_db": summary(np.asarray(values)),
            }

    release_offsets = {
        sequence: {
            scale: summary(
                np.asarray(
                    [reports[sequence][scale][subject]["adaptive"]["geometry"]["offset_from_hard_mean_mm"] for subject in subjects]
                )
            )
            for scale in scales
        }
        for sequence in sequence_paths
    }

    sequence_labels = list(sequence_paths)
    for method in ("adaptive", "free"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        all_values = []
        for scale in scales:
            all_values.extend(
                comparisons[sequence][scale][region][f"{method}_minus_hard"]["mean"]
                for sequence in sequence_labels for region in regions
            )
        limit = max(abs(min(all_values)), abs(max(all_values)))
        for axis, scale in zip(axes, scales):
            matrix = np.asarray(
                [
                    [comparisons[sequence][scale][region][f"{method}_minus_hard"]["mean"] for region in regions]
                    for sequence in sequence_labels
                ]
            )
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
            axis.set_xticks(np.arange(len(regions)), labels=regions)
            axis.set_yticks(np.arange(len(sequence_labels)), labels=sequence_labels)
            axis.set_title(f"{'8 frames' if scale == '8' else 'full scale'}")
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    axis.text(column, row, f"{matrix[row, column]:+.1f}", ha="center", va="center")
        fig.colorbar(image, ax=axes, label=f"{method} - hard PSNR (dB)")
        fig.suptitle(f"Animation intervention across motion families: {method}")
        fig.savefig(args.output / f"multisequence_{method}_minus_hard.png", dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = np.arange(len(sequence_labels))
    width = 0.34
    for offset, scale in ((-width / 2, "8"), (width / 2, "all")):
        means = [localization[sequence][scale]["difference_of_differences_db"]["mean"] for sequence in sequence_labels]
        low = [
            means[index] - localization[sequence][scale]["difference_of_differences_db"]["subject_bootstrap_95ci"][0]
            for index, sequence in enumerate(sequence_labels)
        ]
        high = [
            localization[sequence][scale]["difference_of_differences_db"]["subject_bootstrap_95ci"][1] - means[index]
            for index, sequence in enumerate(sequence_labels)
        ]
        axis.bar(x + offset, means, width, yerr=np.stack([low, high]), capsize=3, label="8 frames" if scale == "8" else "full scale")
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(x, sequence_labels)
    axis.set_ylabel("target-region minus control-region adaptive gain (dB)")
    axis.set_title("Motion-aligned regional interaction")
    axis.legend()
    fig.savefig(args.output / "motion_region_localization.png", dpi=180)
    plt.close(fig)

    result = {
        "kind": "animation_multisequence_region_analysis",
        "subjects": list(subjects),
        "sequences": sequence_labels,
        "scales": list(scales),
        "paired_against_hard": comparisons,
        "motion_aligned_localization": localization,
        "adaptive_offset_from_hard_mm": release_offsets,
        "interpretation": [
            "At eight frames free A is generally worse than hard A, while a bounded adaptive residual is usually beneficial.",
            "At full scale free A dominates whole-head rendering, but adaptive release dominates semantically localized face/oral/eye regions.",
            "The adaptive gain is directionally larger in the motion-relevant region than in its prespecified control for head, eyes, and jaw families; mouth-versus-eye localization is weaker and must be reported rather than hidden.",
        ],
        "scope": "Five identities, four motion families, one public camera, temporal held-out frames, fixed K=16.",
    }
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Animation multi-sequence regional analysis", ""]
    for sequence in sequence_labels:
        item = localization[sequence]["all"]
        stats = item["difference_of_differences_db"]
        lines.append(
            f"- {sequence}: ({item['target_region']} minus {item['control_region']}) adaptive-gain interaction "
            f"{stats['mean']:+.2f} dB, identity bootstrap {stats['subject_bootstrap_95ci']}, {stats['positive_subjects']}/5 positive."
        )
    lines.extend(["", "These are prespecified target-control contrasts; not every family is significant with five identities."])
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "localization_full": {key: value["all"] for key, value in localization.items()}}, indent=2))


if __name__ == "__main__":
    main()
