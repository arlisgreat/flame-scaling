#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from flame_oracle.io import load_multiview_frame, save_montage


ORDER = [0.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, -1.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--subject-root",
        type=Path,
        default=Path("/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs/388"),
    )
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.input.glob("*/metrics.json")]
    reports.sort(key=lambda item: ORDER.index(float(item["radius_mm"])))
    if [float(item["radius_mm"]) for item in reports] != ORDER:
        raise RuntimeError("incomplete or unexpected radius ladder")

    rows = []
    for report in reports:
        evaluation = report["evaluation"]
        geometry = report["geometry"]
        history = report["history"][-1]
        rows.append(
            {
                "radius_mm": report["radius_mm"],
                "label": "free" if report["radius_mm"] < 0 else f"{report['radius_mm']:g}mm",
                "steps": report["steps"],
                "seed": report["seed"],
                "mean_offset_mm": history["mean_offset_mm"],
                "supported_psnr": evaluation["supported"]["masked_psnr"],
                "transition_psnr": evaluation["transition"]["masked_psnr"],
                "unsupported_psnr": evaluation["unsupported"]["masked_psnr"],
                "head_psnr": evaluation["head"]["masked_psnr"],
                "supported_lp_mse": evaluation["supported"]["masked_mse"],
                "unsupported_lp_mse": evaluation["unsupported"]["masked_mse"],
                "pcd_to_means_supported_mean_mm": geometry["pcd_to_means_supported_mean_mm"],
                "pcd_to_means_unsupported_mean_mm": geometry["pcd_to_means_unsupported_mean_mm"],
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for region, style in (("supported", "o-"), ("transition", "s-"), ("unsupported", "^-"), ("head", "D--")):
        axes[0].plot(x, [row[f"{region}_psnr"] for row in rows], style, label=region)
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Maximum geometry release")
    axes[0].set_ylabel("Masked PSNR (dB)")
    axes[0].set_title("Multiview oracle fit")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(x, [row["pcd_to_means_supported_mean_mm"] for row in rows], "o-", label="supported")
    axes[1].plot(x, [row["pcd_to_means_unsupported_mean_mm"] for row in rows], "^-", label="unsupported")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Maximum geometry release")
    axes[1].set_ylabel("GT PCD to Gaussian mean (mm)")
    axes[1].set_title("Geometry coverage")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output / "constraint_curve.png", dpi=180)
    plt.close(fig)

    reference = reports[0]
    height = int(reference["resolution"][0])
    case = load_multiview_frame(args.subject_root, "GLASSES", 0, height)
    selected = [0, min(6, len(case["camera_ids"]) - 1), len(case["camera_ids"]) - 1]
    images = []
    for camera_index in selected:
        camera_id = case["camera_ids"][camera_index]
        entries = [
            ("target", case["images"][camera_index]),
            ("hard", cv2.cvtColor(cv2.imread(str(args.input / "r0mm" / f"render_{camera_id}.png")), cv2.COLOR_BGR2RGB) / 255.0),
            ("25mm", cv2.cvtColor(cv2.imread(str(args.input / "r25mm" / f"render_{camera_id}.png")), cv2.COLOR_BGR2RGB) / 255.0),
            ("200mm", cv2.cvtColor(cv2.imread(str(args.input / "r200mm" / f"render_{camera_id}.png")), cv2.COLOR_BGR2RGB) / 255.0),
            ("free", cv2.cvtColor(cv2.imread(str(args.input / "free" / f"render_{camera_id}.png")), cv2.COLOR_BGR2RGB) / 255.0),
        ]
        for label, image in entries:
            annotated = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            cv2.putText(annotated, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 0), 1, cv2.LINE_AA)
            images.append(annotated)
    save_montage(args.output / "qualitative_montage.png", images, columns=5)

    hard, wide, free = rows[0], rows[-2], rows[-1]
    markdown = f"""# Geometry oracle pilot: subject 388 / GLASSES / frame 0

This is a single-case, single-seed oracle diagnostic, not a population-level result.

- Common budget: {reference['gaussian_count']} Gaussians, {reference['steps']} steps, {height} px image height, {len(reference['camera_ids'])} calibrated views.
- Hard FLAME supported/unsupported PSNR: {hard['supported_psnr']:.2f} / {hard['unsupported_psnr']:.2f} dB.
- Free geometry supported/unsupported PSNR: {free['supported_psnr']:.2f} / {free['unsupported_psnr']:.2f} dB.
- Free-minus-hard gain: {free['supported_psnr'] - hard['supported_psnr']:+.2f} dB supported, {free['unsupported_psnr'] - hard['unsupported_psnr']:+.2f} dB unsupported, {free['head_psnr'] - hard['head_psnr']:+.2f} dB whole head.
- Unsupported GT-to-mean geometry error: {hard['pcd_to_means_unsupported_mean_mm']:.2f} -> {free['pcd_to_means_unsupported_mean_mm']:.2f} mm.
- Supported GT-to-mean geometry error: {hard['pcd_to_means_supported_mean_mm']:.2f} -> {free['pcd_to_means_supported_mean_mm']:.2f} mm.
- 200mm-minus-free gap: {wide['supported_psnr'] - free['supported_psnr']:+.2f} dB supported, {wide['unsupported_psnr'] - free['unsupported_psnr']:+.2f} dB unsupported, {wide['head_psnr'] - free['head_psnr']:+.2f} dB whole head.

Interpretation: tight geometry constraints create a clear ceiling in hair/glasses/beard-like regions, but a 200mm bounded offset closes the gap to free geometry in this static case. Therefore this run does **not** establish that FLAME query initialization itself is the bottleneck for methods that already allow approximately 200mm offsets. It redirects the dynamic hypothesis toward correspondence and animation binding. The result must still be repeated across seeds, subjects, regions, resolutions, Gaussian counts, and optimization budgets.
"""
    (args.output / "report.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
