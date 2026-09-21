#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from flame_oracle.flame import load_flame_linear_assets
from flame_oracle.io import save_montage


DEFAULT_ASSET_ROOT = Path(
    "/data1/workspace/airulan/3dv/external/LAM/model_zoo/human_parametric_models/flame_vhap"
)


def bootstrap_interval(values: np.ndarray, samples: int = 20000) -> list[float]:
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(0)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def summarize(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()),
        "subject_bootstrap_95ci": bootstrap_interval(array),
        "per_subject": array.tolist(),
        "positive_subjects": int((array > 0).sum()),
        "subject_count": len(array),
    }


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/runs/animation_multisubject_exp5"))
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    subjects = ("393", "404", "461", "477", "486")
    scale_labels = ("8", "24", "all")
    methods = ("hard", "adaptive", "free")
    regions = ("head", "face", "oral", "eyes")
    reports = {
        scale: {
            subject: {
                method: json.loads((args.root / f"n{scale}" / subject / method / "metrics.json").read_text())
                for method in methods
            }
            for subject in subjects
        }
        for scale in scale_labels
    }
    full_counts = [reports["all"][subject]["hard"]["train_count"] for subject in subjects]

    curves = {
        region: {
            method: {
                scale: [
                    reports[scale][subject][method]["evaluation"][region]["heldout"]["masked_psnr"]
                    for subject in subjects
                ]
                for scale in scale_labels
            }
            for method in methods
        }
        for region in regions
    }
    paired = {}
    for scale in scale_labels:
        paired[scale] = {}
        for region in regions:
            paired[scale][region] = {}
            for comparison, left, right in (
                ("adaptive_minus_hard", "adaptive", "hard"),
                ("free_minus_hard", "free", "hard"),
                ("adaptive_minus_free", "adaptive", "free"),
            ):
                values = [
                    reports[scale][subject][left]["evaluation"][region]["heldout"]["masked_psnr"]
                    - reports[scale][subject][right]["evaluation"][region]["heldout"]["masked_psnr"]
                    for subject in subjects
                ]
                paired[scale][region][comparison] = summarize(values)

    scale_gain = {}
    for region in regions:
        scale_gain[region] = {}
        for method in methods:
            values = [
                reports["all"][subject][method]["evaluation"][region]["heldout"]["masked_psnr"]
                - reports["8"][subject][method]["evaluation"][region]["heldout"]["masked_psnr"]
                for subject in subjects
            ]
            scale_gain[region][method] = summarize(values)

    colors = {"hard": "#444444", "adaptive": "#e76f51", "free": "#457b9d"}
    labels = {"hard": "hard A", "adaptive": "adaptive release", "free": "free A"}
    x = np.arange(len(scale_labels))
    xlabels = ["8", "24", f"full\n({min(full_counts)}-{max(full_counts)})"]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, region in zip(axes.ravel(), regions):
        for method in methods:
            values = np.asarray([curves[region][method][scale] for scale in scale_labels])
            mean = values.mean(axis=1)
            low, high = [], []
            for row in values:
                interval = bootstrap_interval(row)
                low.append(mean[len(low)] - interval[0])
                high.append(interval[1] - mean[len(high)])
            axis.errorbar(
                x, mean, yerr=np.stack([low, high]), marker="o", capsize=3,
                color=colors[method], label=labels[method], linewidth=2,
            )
        axis.set_xticks(x, xlabels)
        axis.set_title(region)
        axis.set_xlabel("training frames")
        axis.set_ylabel("held-out temporal PSNR (dB)")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.output / "multisubject_animation_scale.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, region in zip(axes, ("head", "oral")):
        for subject in subjects:
            values = [
                reports[scale][subject]["free"]["evaluation"][region]["heldout"]["masked_psnr"]
                - reports[scale][subject]["hard"]["evaluation"][region]["heldout"]["masked_psnr"]
                for scale in scale_labels
            ]
            axis.plot(x, values, marker="o", alpha=0.55, label=subject)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(x, xlabels)
        axis.set_title(f"{region}: free A - hard A")
        axis.set_ylabel("paired held-out PSNR difference (dB)")
        axis.set_xlabel("training frames")
    axes[0].legend(title="subject", fontsize=7)
    fig.savefig(args.output / "subject_crossover_spaghetti.png", dpi=180)
    plt.close(fig)

    assets = load_flame_linear_assets(
        args.asset_root / "flame2023.pkl", args.asset_root / "FLAME_masks.pkl", 300, 100
    )
    gate_regions = {}
    for scale in scale_labels:
        gate_regions[scale] = {}
        for region in ("face", "lips", "eye_region", "scalp", "neck"):
            values = []
            for subject in subjects:
                gates = np.load(args.root / f"n{scale}" / subject / "adaptive" / "model.npz")["release_gates"]
                values.append(float(gates[assets["masks"][region]].mean()))
            gate_regions[scale][region] = summarize(values)

    qualitative = []
    for subject in subjects:
        frame_names = sorted((args.root / "nall" / subject / "hard").glob("heldout_*.png"))
        frame_name = frame_names[len(frame_names) // 2].name
        for method in methods:
            qualitative.append(read_rgb(args.root / "nall" / subject / method / frame_name))
    save_montage(args.output / "fullscale_qualitative_methods_by_subject.png", qualitative, columns=3)

    headline = {
        "oral_free_minus_hard_n8": paired["8"]["oral"]["free_minus_hard"],
        "oral_free_minus_hard_full": paired["all"]["oral"]["free_minus_hard"],
        "oral_adaptive_minus_hard_full": paired["all"]["oral"]["adaptive_minus_hard"],
        "oral_adaptive_minus_free_full": paired["all"]["oral"]["adaptive_minus_free"],
        "head_free_minus_adaptive_full": {
            **summarize(
                [
                    reports["all"][subject]["free"]["evaluation"]["head"]["heldout"]["masked_psnr"]
                    - reports["all"][subject]["adaptive"]["evaluation"]["head"]["heldout"]["masked_psnr"]
                    for subject in subjects
                ]
            )
        },
        "oral_scale_gain_hard": scale_gain["oral"]["hard"],
        "oral_scale_gain_adaptive": scale_gain["oral"]["adaptive"],
        "oral_scale_gain_free": scale_gain["oral"]["free"],
    }
    result = {
        "kind": "multisubject_animation_scale_analysis",
        "subjects": list(subjects),
        "scale_labels": list(scale_labels),
        "full_train_counts": dict(zip(subjects, full_counts)),
        "curves": curves,
        "paired_subject_statistics": paired,
        "paired_scale_gain": scale_gain,
        "adaptive_release_gate_by_flame_region": gate_regions,
        "headline": headline,
        "interpretation": [
            "At eight frames hard A beats free A in oral detail for every identity.",
            "At full scale free A beats hard A in oral detail for every identity: the sign reversal replicates across subjects.",
            "Adaptive release beats both endpoints in full-scale oral detail for every identity.",
            "At full scale free A is better for the whole head while adaptive release is better for oral detail, so prior strength should be region dependent.",
        ],
        "scope": "Five identities, one mouth-motion sequence per identity, temporal held-out frames from the public training camera.",
    }
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Multi-subject animation-scale result",
        "",
        f"- Oral free-hard at 8 frames: {headline['oral_free_minus_hard_n8']['mean']:+.2f} dB, identity bootstrap {headline['oral_free_minus_hard_n8']['subject_bootstrap_95ci']} ({headline['oral_free_minus_hard_n8']['positive_subjects']}/5 positive).",
        f"- Oral free-hard at full scale: {headline['oral_free_minus_hard_full']['mean']:+.2f} dB, identity bootstrap {headline['oral_free_minus_hard_full']['subject_bootstrap_95ci']} ({headline['oral_free_minus_hard_full']['positive_subjects']}/5 positive).",
        f"- Oral adaptive-hard at full scale: {headline['oral_adaptive_minus_hard_full']['mean']:+.2f} dB, identity bootstrap {headline['oral_adaptive_minus_hard_full']['subject_bootstrap_95ci']}.",
        f"- Oral adaptive-free at full scale: {headline['oral_adaptive_minus_free_full']['mean']:+.2f} dB, identity bootstrap {headline['oral_adaptive_minus_free_full']['subject_bootstrap_95ci']}.",
        f"- Head free-adaptive at full scale: {headline['head_free_minus_adaptive_full']['mean']:+.2f} dB, identity bootstrap {headline['head_free_minus_adaptive_full']['subject_bootstrap_95ci']}.",
        "",
        "Intervals resample identities, not frames.",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "headline": headline}, indent=2))


if __name__ == "__main__":
    main()
