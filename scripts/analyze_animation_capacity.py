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
    parser.add_argument("--capacity-root", type=Path, default=Path("artifacts/runs/animation_capacity_exp5"))
    parser.add_argument("--baseline-root", type=Path, default=Path("artifacts/runs/animation_multisubject_exp5"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    subjects = ("393", "404", "461", "477", "486")
    scales = ("8", "all")
    capacities = (4, 8, 16, 32, 64)
    methods = ("adaptive", "free")
    regions = ("head", "face", "oral", "eyes")
    reports: dict[str, dict[int, dict[str, dict[str, dict[str, object]]]]] = {}
    hard_reports = {}
    for scale in scales:
        hard_reports[scale] = {
            subject: json.loads((args.baseline_root / f"n{scale}" / subject / "hard" / "metrics.json").read_text())
            for subject in subjects
        }
        reports[scale] = {}
        for capacity in capacities:
            reports[scale][capacity] = {}
            for subject in subjects:
                reports[scale][capacity][subject] = {}
                for method in methods:
                    if capacity == 16:
                        path = args.baseline_root / f"n{scale}" / subject / method / "metrics.json"
                    else:
                        path = args.capacity_root / f"n{scale}" / subject / f"{method}_k{capacity}" / "metrics.json"
                    reports[scale][capacity][subject][method] = json.loads(path.read_text())

    metrics = {}
    comparisons = {}
    for scale in scales:
        metrics[scale] = {}
        comparisons[scale] = {}
        for capacity in capacities:
            metrics[scale][str(capacity)] = {}
            comparisons[scale][str(capacity)] = {}
            for region in regions:
                metrics[scale][str(capacity)][region] = {}
                comparisons[scale][str(capacity)][region] = {}
                hard = np.asarray(
                    [hard_reports[scale][subject]["evaluation"][region]["heldout"]["masked_psnr"] for subject in subjects]
                )
                for method in methods:
                    held = np.asarray(
                        [reports[scale][capacity][subject][method]["evaluation"][region]["heldout"]["masked_psnr"] for subject in subjects]
                    )
                    train = np.asarray(
                        [reports[scale][capacity][subject][method]["evaluation"][region]["train"]["masked_psnr"] for subject in subjects]
                    )
                    metrics[scale][str(capacity)][region][method] = {
                        "heldout_psnr": summary(held),
                        "train_psnr": summary(train),
                        "generalization_gap_db": summary(train - held),
                    }
                    comparisons[scale][str(capacity)][region][f"{method}_minus_hard"] = summary(held - hard)

    best_capacity = {
        scale: {
            region: {
                method: max(
                    capacities,
                    key=lambda capacity: metrics[scale][str(capacity)][region][method]["heldout_psnr"]["mean"],
                )
                for method in methods
            }
            for region in regions
        }
        for scale in scales
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"hard": "#444444", "adaptive": "#e76f51", "free": "#457b9d"}
    labels = {"hard": "hard A", "adaptive": "adaptive release", "free": "free A"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    for column, scale in enumerate(scales):
        axis = axes[0, column]
        hard = np.mean(
            [hard_reports[scale][subject]["evaluation"]["oral"]["heldout"]["masked_psnr"] for subject in subjects]
        )
        axis.axhline(hard, color=colors["hard"], label=labels["hard"], linewidth=2)
        for method in methods:
            means, lows, highs = [], [], []
            for capacity in capacities:
                item = metrics[scale][str(capacity)]["oral"][method]["heldout_psnr"]
                means.append(item["mean"])
                lows.append(item["mean"] - item["subject_bootstrap_95ci"][0])
                highs.append(item["subject_bootstrap_95ci"][1] - item["mean"])
            axis.errorbar(capacities, means, yerr=np.stack([lows, highs]), marker="o", capsize=3, color=colors[method], label=labels[method])
        axis.set_xscale("log", base=2)
        axis.set_xticks(capacities, labels=[str(x) for x in capacities])
        axis.set_title(f"oral held-out, {'8 frames' if scale == '8' else 'full scale'}")
        axis.set_ylabel("PSNR (dB)")
        axis.set_xlabel("deformation latent capacity K")
        axis.legend(fontsize=8)

        gap_axis = axes[1, column]
        for method in methods:
            means = [metrics[scale][str(capacity)]["oral"][method]["generalization_gap_db"]["mean"] for capacity in capacities]
            gap_axis.plot(capacities, means, marker="o", color=colors[method], label=labels[method])
        gap_axis.set_xscale("log", base=2)
        gap_axis.set_xticks(capacities, labels=[str(x) for x in capacities])
        gap_axis.set_title(f"oral generalization gap, {'8 frames' if scale == '8' else 'full scale'}")
        gap_axis.set_ylabel("train-held-out PSNR (dB)")
        gap_axis.set_xlabel("deformation latent capacity K")
        gap_axis.legend(fontsize=8)
    fig.savefig(args.output / "capacity_quality_and_gap.png", dpi=180)
    plt.close(fig)

    advantage = np.zeros((4, len(capacities)), np.float32)
    row_labels = []
    for row, (scale, method) in enumerate((("8", "adaptive"), ("8", "free"), ("all", "adaptive"), ("all", "free"))):
        advantage[row] = [comparisons[scale][str(capacity)]["oral"][f"{method}_minus_hard"]["mean"] for capacity in capacities]
        row_labels.append(f"{'8' if scale == '8' else 'full'} / {method}")
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    limit = float(np.max(np.abs(advantage)))
    image = axis.imshow(advantage, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(capacities)), labels=[str(x) for x in capacities])
    axis.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    axis.set_xlabel("deformation latent capacity K")
    axis.set_title("oral held-out advantage over hard A (dB)")
    for row in range(advantage.shape[0]):
        for column in range(advantage.shape[1]):
            axis.text(column, row, f"{advantage[row, column]:+.1f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, label="PSNR difference (dB)")
    fig.savefig(args.output / "capacity_scale_interaction_heatmap.png", dpi=180)
    plt.close(fig)

    headline = {
        "best_capacity": best_capacity,
        "n8_oral_free_k4_minus_k64_db": float(
            metrics["8"]["4"]["oral"]["free"]["heldout_psnr"]["mean"]
            - metrics["8"]["64"]["oral"]["free"]["heldout_psnr"]["mean"]
        ),
        "n8_oral_free_gap_k4_db": metrics["8"]["4"]["oral"]["free"]["generalization_gap_db"],
        "n8_oral_free_gap_k64_db": metrics["8"]["64"]["oral"]["free"]["generalization_gap_db"],
        "full_oral_adaptive_best": metrics["all"][str(best_capacity["all"]["oral"]["adaptive"])]["oral"]["adaptive"]["heldout_psnr"],
        "full_oral_free_best": metrics["all"][str(best_capacity["all"]["oral"]["free"])]["oral"]["free"]["heldout_psnr"],
    }
    result = {
        "kind": "animation_capacity_scale_interaction",
        "subjects": list(subjects),
        "capacities": list(capacities),
        "scales": list(scales),
        "metrics": metrics,
        "paired_against_hard": comparisons,
        "best_capacity_by_scale_region_method": best_capacity,
        "headline": headline,
        "interpretation": [
            "At eight frames, increasing learned animation capacity monotonically worsens free-A held-out oral quality and increases the train-test gap.",
            "At full scale, both adaptive and free animation have an intermediate-capacity optimum rather than a monotonic capacity law.",
            "Adaptive release stays above hard A over a broader capacity range; free A needs both enough data and calibrated capacity.",
        ],
        "scope": "Five identities, mouth-motion sequence, temporal held-out frames; capacity is the PCA-conditioned deformation rank K.",
    }
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Animation capacity x scale",
        "",
        f"- At 8 frames, free-A oral K=4 exceeds K=64 by {headline['n8_oral_free_k4_minus_k64_db']:+.2f} dB.",
        f"- Free-A oral train-test gap grows from {headline['n8_oral_free_gap_k4_db']['mean']:.2f} dB (K=4) to {headline['n8_oral_free_gap_k64_db']['mean']:.2f} dB (K=64).",
        f"- Full-scale best oral capacity: adaptive K={best_capacity['all']['oral']['adaptive']}, free K={best_capacity['all']['oral']['free']}.",
        "",
        "Capacity is not a monotonic proxy for representation ceiling; it interacts with data through estimation error.",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "headline": headline}, indent=2))


if __name__ == "__main__":
    main()
