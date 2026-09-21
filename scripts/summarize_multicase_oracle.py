#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from flame_oracle.io import load_multiview_frame, save_montage


METHODS = [("r0mm", "hard"), ("r200mm", "200mm"), ("free", "free")]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q * 100.0))


def bootstrap_mean(values: list[float], replicates: int = 10000, seed: int = 20260715) -> dict[str, float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        samples.append(float(np.mean(rng.choices(values, k=len(values)))))
    return {
        "mean": float(np.mean(values)),
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("configs/oracle_cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.cases.read_text())
    rows = []
    reports = {}
    for case in config["cases"]:
        subject = case["subject"]
        reports[subject] = {}
        for folder, method in METHODS:
            report = json.loads((args.input / subject / folder / "metrics.json").read_text())
            reports[subject][method] = report
            row = {
                "subject": subject,
                "sequence": case["sequence"],
                "visible_stress_region": case["stress_region"],
                "method": method,
                "radius_mm": report["radius_mm"],
                "steps": report["steps"],
                "total_trainable_parameters": report["total_trainable_parameters"],
                "active_position_parameters": report["active_position_parameters"],
                "supported_psnr": report["evaluation"]["supported"]["masked_psnr"],
                "unsupported_psnr": report["evaluation"]["unsupported"]["masked_psnr"],
                "head_psnr": report["evaluation"]["head"]["masked_psnr"],
                "unsupported_geometry_mm": report["geometry"]["pcd_to_means_unsupported_mean_mm"],
                "supported_geometry_mm": report["geometry"]["pcd_to_means_supported_mean_mm"],
                "final_loss": report["history"][-1]["loss"],
            }
            rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_case.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    effects = {}
    for comparison, left, right in (("free_minus_hard", "free", "hard"), ("free_minus_200mm", "free", "200mm")):
        effects[comparison] = {}
        for metric in ("supported_psnr", "unsupported_psnr", "head_psnr"):
            values = []
            per_subject = {}
            for case in config["cases"]:
                subject = case["subject"]
                by_method = {row["method"]: row for row in rows if row["subject"] == subject}
                value = float(by_method[left][metric] - by_method[right][metric])
                values.append(value)
                per_subject[subject] = value
            effects[comparison][metric] = {**bootstrap_mean(values), "per_subject": per_subject}

    geometry_gain = []
    for case in config["cases"]:
        subject = case["subject"]
        by_method = {row["method"]: row for row in rows if row["subject"] == subject}
        geometry_gain.append(
            float(by_method["hard"]["unsupported_geometry_mm"] - by_method["free"]["unsupported_geometry_mm"])
        )
    effects["hard_minus_free_unsupported_geometry_mm"] = bootstrap_mean(geometry_gain)
    (args.output / "paired_effects.json").write_text(json.dumps(effects, indent=2) + "\n")

    subjects = [case["subject"] for case in config["cases"]]
    x = np.arange(len(subjects))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {"hard": "#d62728", "200mm": "#ff7f0e", "free": "#1f77b4"}
    for method in ("hard", "200mm", "free"):
        values = [next(row["unsupported_psnr"] for row in rows if row["subject"] == subject and row["method"] == method) for subject in subjects]
        axes[0].plot(x, values, "o-", label=method, color=colors[method])
    axes[0].set_xticks(x, subjects)
    axes[0].set_ylabel("Unsupported-region PSNR (dB)")
    axes[0].set_title("Converged oracle fit")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for metric, marker in (("supported_psnr", "o"), ("unsupported_psnr", "^"), ("head_psnr", "s")):
        values = [effects["free_minus_200mm"][metric]["per_subject"][subject] for subject in subjects]
        axes[1].plot(x, values, marker + "-", label=metric.removesuffix("_psnr"))
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axhspan(-0.75, 0.75, color="gray", alpha=0.15, label="exploratory ±0.75 dB")
    axes[1].set_xticks(x, subjects)
    axes[1].set_ylabel("Free minus 200mm (dB)")
    axes[1].set_title("Wide release vs free")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    hard_geometry = [next(row["unsupported_geometry_mm"] for row in rows if row["subject"] == subject and row["method"] == "hard") for subject in subjects]
    free_geometry = [next(row["unsupported_geometry_mm"] for row in rows if row["subject"] == subject and row["method"] == "free") for subject in subjects]
    axes[2].plot(x, hard_geometry, "o-", label="hard", color=colors["hard"])
    axes[2].plot(x, free_geometry, "o-", label="free", color=colors["free"])
    axes[2].set_xticks(x, subjects)
    axes[2].set_ylabel("Unsupported GT-to-mean (mm)")
    axes[2].set_title("Geometry coverage")
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(args.output / "multicase_effects.png", dpi=180)
    plt.close(fig)

    montage = []
    dataset_root = Path(config["dataset_root"])
    for case in config["cases"]:
        subject = case["subject"]
        reference = reports[subject]["hard"]
        height = int(reference["resolution"][0])
        frame = load_multiview_frame(dataset_root / subject, case["sequence"], case["frame"], height)
        camera_id = "222200041" if "222200041" in frame["camera_ids"] else frame["camera_ids"][0]
        camera_index = frame["camera_ids"].index(camera_id)
        entries = [("target", frame["images"][camera_index])]
        for folder, method in METHODS:
            path = args.input / subject / folder / f"render_{camera_id}.png"
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            entries.append((method, image))
        for label, image in entries:
            annotated = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            cv2.putText(annotated, f"{subject} {label}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 0, 0), 1, cv2.LINE_AA)
            montage.append(annotated)
    save_montage(args.output / "qualitative_multicase.png", montage, columns=4)

    hard_effect = effects["free_minus_hard"]
    wide_effect = effects["free_minus_200mm"]
    geometry_effect = effects["hard_minus_free_unsupported_geometry_mm"]
    report = f"""# Multi-case geometry oracle findings

## Evidence scope

- Five NeRSemble NVS subjects, 13 calibrated views per subject, one GT point cloud per case.
- 5,023 Gaussians and 50,230 total trainable parameters in every method.
- 4,000 optimization steps at 128 px image height; all methods use the same initial world-coordinate position LR.
- Hard has the same dormant position parameter tensor but zero active geometry DoF.

## Finding G1: tight FLAME geometry is a real static representation ceiling

- Free minus hard supported-region PSNR: {hard_effect['supported_psnr']['mean']:+.2f} dB, bootstrap 95% CI [{hard_effect['supported_psnr']['ci95_low']:+.2f}, {hard_effect['supported_psnr']['ci95_high']:+.2f}].
- Free minus hard unsupported-region PSNR: {hard_effect['unsupported_psnr']['mean']:+.2f} dB, bootstrap 95% CI [{hard_effect['unsupported_psnr']['ci95_low']:+.2f}, {hard_effect['unsupported_psnr']['ci95_high']:+.2f}].
- Free minus hard whole-head PSNR: {hard_effect['head_psnr']['mean']:+.2f} dB, bootstrap 95% CI [{hard_effect['head_psnr']['ci95_low']:+.2f}, {hard_effect['head_psnr']['ci95_high']:+.2f}].
- Hard minus free unsupported geometry error: {geometry_effect['mean']:+.2f} mm, bootstrap 95% CI [{geometry_effect['ci95_low']:+.2f}, {geometry_effect['ci95_high']:+.2f}].

## Finding G2: 200mm release is practically close to free geometry in this static oracle

- Free minus 200mm supported PSNR: {wide_effect['supported_psnr']['mean']:+.2f} dB, bootstrap 95% CI [{wide_effect['supported_psnr']['ci95_low']:+.2f}, {wide_effect['supported_psnr']['ci95_high']:+.2f}].
- Free minus 200mm unsupported PSNR: {wide_effect['unsupported_psnr']['mean']:+.2f} dB, bootstrap 95% CI [{wide_effect['unsupported_psnr']['ci95_low']:+.2f}, {wide_effect['unsupported_psnr']['ci95_high']:+.2f}].
- Free minus 200mm whole-head PSNR: {wide_effect['head_psnr']['mean']:+.2f} dB, bootstrap 95% CI [{wide_effect['head_psnr']['ci95_low']:+.2f}, {wide_effect['head_psnr']['ci95_high']:+.2f}].

The ±0.75 dB band in the figure is exploratory, not preregistered equivalence testing. The result says that FLAME geometry can be a bottleneck when kept tight, but FLAME query initialization is not itself a static ceiling once approximately 200mm release is available.

## What this does not establish

- The distance bands are FLAME-distance envelopes, not semantic ground-truth masks.
- All views are used for both oracle fitting and evaluation by design; this is not generalization evidence.
- Frame 0 of subject 445's tongue sequence does not visibly expose the tongue, so this run is not tongue evidence.
- This experiment does not release FLAME-semantic correspondence, LBS/blendshape animation, or tracker labels. Those are the next causal targets.
"""
    (args.output / "findings.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

