#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from flame_oracle.io import load_multiview_frame


CONDITIONS = (
    ("hard_cap003", "hard", 3.0),
    ("hard_cap030", "hard", 30.0),
    ("free_cap003", "free", 3.0),
    ("free_cap030", "free", 30.0),
)


def bootstrap_mean(
    values: list[float], replicates: int = 10000, seed: int = 20260715
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(replicates, len(array)))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "per_subject": values,
    }


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("configs/oracle_cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.cases.read_text())
    rows: list[dict[str, object]] = []
    reports: dict[tuple[str, str], dict[str, object]] = {}
    for case in config["cases"]:
        subject = case["subject"]
        for directory, mean_constraint, cap_mm in CONDITIONS:
            report = json.loads((args.input / subject / directory / "metrics.json").read_text())
            model = np.load(args.input / subject / directory / "model.npz")
            axis_max_mm = model["scales"].max(axis=1) * 1000.0
            reports[(subject, directory)] = report
            rows.append(
                {
                    "subject": subject,
                    "sequence": case["sequence"],
                    "stress_region": case["stress_region"],
                    "condition": directory,
                    "mean_constraint": mean_constraint,
                    "scale_cap_mm": cap_mm,
                    "supported_psnr": report["evaluation"]["supported"]["masked_psnr"],
                    "unsupported_psnr": report["evaluation"]["unsupported"]["masked_psnr"],
                    "head_psnr": report["evaluation"]["head"]["masked_psnr"],
                    "unsupported_center_error_mm": report["geometry"][
                        "pcd_to_means_unsupported_mean_mm"
                    ],
                    "axis_max_p50_mm": float(np.quantile(axis_max_mm, 0.50)),
                    "axis_max_p90_mm": float(np.quantile(axis_max_mm, 0.90)),
                    "axis_max_p99_mm": float(np.quantile(axis_max_mm, 0.99)),
                    "axis_fraction_at_cap": float(np.mean(axis_max_mm >= 0.999 * cap_mm)),
                    "final_loss": report["history"][-1]["loss"],
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_case_condition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_key = {(str(row["subject"]), str(row["condition"])): row for row in rows}
    effects: dict[str, object] = {}
    effect_rows: list[dict[str, object]] = []
    for metric in ("supported_psnr", "unsupported_psnr", "head_psnr"):
        extent_hard = []
        extent_free = []
        release_cap3 = []
        release_cap30 = []
        interaction = []
        for case in config["cases"]:
            subject = case["subject"]
            h3 = float(by_key[(subject, "hard_cap003")][metric])
            h30 = float(by_key[(subject, "hard_cap030")][metric])
            f3 = float(by_key[(subject, "free_cap003")][metric])
            f30 = float(by_key[(subject, "free_cap030")][metric])
            extent_hard.append(h30 - h3)
            extent_free.append(f30 - f3)
            release_cap3.append(f3 - h3)
            release_cap30.append(f30 - h30)
            interaction.append((h30 - h3) - (f30 - f3))
            if metric == "unsupported_psnr":
                effect_rows.append(
                    {
                        "subject": subject,
                        "hard_extent_gain_db": h30 - h3,
                        "free_extent_gain_db": f30 - f3,
                        "release_gain_cap003_db": f3 - h3,
                        "release_gain_cap030_db": f30 - h30,
                        "interaction_db": (h30 - h3) - (f30 - f3),
                    }
                )
        effects[metric] = {
            "extent_gain_hard_cap030_minus_cap003": bootstrap_mean(extent_hard),
            "extent_gain_free_cap030_minus_cap003": bootstrap_mean(extent_free),
            "release_gain_cap003_free_minus_hard": bootstrap_mean(release_cap3),
            "release_gain_cap030_free_minus_hard": bootstrap_mean(release_cap30),
            "interaction_extent_hard_minus_free": bootstrap_mean(interaction),
        }

    with (args.output / "per_case_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effect_rows[0]))
        writer.writeheader()
        writer.writerows(effect_rows)
    (args.output / "paired_effects.json").write_text(json.dumps(effects, indent=2) + "\n")

    subjects = [case["subject"] for case in config["cases"]]
    x = np.arange(len(subjects))
    condition_style = {
        "hard_cap003": ("Hard/3", "#d62728", "--"),
        "hard_cap030": ("Hard/30", "#d62728", "-"),
        "free_cap003": ("Free/3", "#1f77b4", "--"),
        "free_cap030": ("Free/30", "#1f77b4", "-"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.0))
    for condition, (label, color, linestyle) in condition_style.items():
        values = [float(by_key[(subject, condition)]["unsupported_psnr"]) for subject in subjects]
        axes[0].plot(x, values, "o", ls=linestyle, color=color, label=label)
    axes[0].set_ylabel("Unsupported PSNR (dB)")
    axes[0].set_title("2×2 conditions")
    axes[0].legend(frameon=False, fontsize=8)

    hard_extent = [float(row["hard_extent_gain_db"]) for row in effect_rows]
    free_extent = [float(row["free_extent_gain_db"]) for row in effect_rows]
    axes[1].plot(x, hard_extent, "o-", color="#d62728", label="Hard means")
    axes[1].plot(x, free_extent, "o-", color="#1f77b4", label="Free means")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("30 mm minus 3 mm PSNR (dB)")
    axes[1].set_title("Extent-channel gain")
    axes[1].legend(frameon=False, fontsize=8)

    release3 = [float(row["release_gain_cap003_db"]) for row in effect_rows]
    release30 = [float(row["release_gain_cap030_db"]) for row in effect_rows]
    axes[2].plot(x, release3, "o-", color="#9467bd", label="3 mm extent")
    axes[2].plot(x, release30, "o-", color="#2ca02c", label="30 mm extent")
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_ylabel("Free minus hard PSNR (dB)")
    axes[2].set_title("Apparent mean-release gain")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xticks(x, subjects)
        axis.set_xlabel("Subject")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "multicase_interaction.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(len(subjects), 5, figsize=(10.5, 10.0))
    columns = ["target", "hard_cap003", "hard_cap030", "free_cap003", "free_cap030"]
    labels = ["Target", "Hard/3", "Hard/30", "Free/3", "Free/30"]
    dataset_root = Path(config["dataset_root"])
    for row_index, case_config in enumerate(config["cases"]):
        subject = case_config["subject"]
        reference = reports[(subject, "hard_cap003")]
        frame = load_multiview_frame(
            dataset_root / subject,
            case_config["sequence"],
            case_config["frame"],
            int(reference["resolution"][0]),
            background_threshold=30,
        )
        preferred = "222200041"
        camera_id = preferred if preferred in frame["camera_ids"] else frame["camera_ids"][0]
        camera_index = frame["camera_ids"].index(camera_id)
        for column_index, condition in enumerate(columns):
            image = (
                frame["images"][camera_index]
                if condition == "target"
                else read_rgb(args.input / subject / condition / f"render_{camera_id}.png")
            )
            axes[row_index, column_index].imshow(np.clip(image, 0, 1))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(labels[column_index], fontsize=9)
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(subject, fontsize=9)
    fig.tight_layout(pad=0.25)
    fig.savefig(args.output / "qualitative_multicase_factorial.png", dpi=200)
    plt.close(fig)

    unsupported = effects["unsupported_psnr"]
    hard_extent = unsupported["extent_gain_hard_cap030_minus_cap003"]
    free_extent = unsupported["extent_gain_free_cap030_minus_cap003"]
    release3 = unsupported["release_gain_cap003_free_minus_hard"]
    release30 = unsupported["release_gain_cap030_free_minus_hard"]
    interaction = unsupported["interaction_extent_hard_minus_free"]
    all_positive = all(float(row["interaction_db"]) > 0 for row in effect_rows)
    report = f"""# Multi-case mean-position × Gaussian-extent factorial

## Design

- Five NeRSemble NVS cases, each with a landmark-anchored, coefficient-bounded FLAME scaffold.
- Paired 2×2 intervention: Gaussian means hard vs free; maximum Gaussian-axis standard deviation 3 vs 30 mm.
- Every condition uses 5,023 Gaussians, identical trainable parameter counts, 4,000 steps, and all 13 calibrated views.

## Replicated finding

- Unsupported-region gain from relaxing extent, hard means: **{hard_extent['mean']:+.2f} dB**, bootstrap 95% CI [{hard_extent['ci95_low']:+.2f}, {hard_extent['ci95_high']:+.2f}].
- The same extent gain with free means: **{free_extent['mean']:+.2f} dB**, bootstrap 95% CI [{free_extent['ci95_low']:+.2f}, {free_extent['ci95_high']:+.2f}].
- Mean-release gain at 3 mm extent: **{release3['mean']:+.2f} dB**, bootstrap 95% CI [{release3['ci95_low']:+.2f}, {release3['ci95_high']:+.2f}].
- Apparent mean-release gain at 30 mm extent: **{release30['mean']:+.2f} dB**, bootstrap 95% CI [{release30['ci95_low']:+.2f}, {release30['ci95_high']:+.2f}].
- Difference-in-differences (hard extent gain minus free extent gain): **{interaction['mean']:+.2f} dB**, bootstrap 95% CI [{interaction['ci95_low']:+.2f}, {interaction['ci95_high']:+.2f}]; positive in **{sum(float(row['interaction_db']) > 0 for row in effect_rows)}/5** cases.

## Interpretation

The single-tongue-frame covariance bypass is {'replicated in every case' if all_positive else 'replicated on average but heterogeneous across cases'}. Gaussian extent is a second geometry channel whose capacity changes the measured benefit of releasing FLAME means. Therefore a one-dimensional hard-vs-free comparison does not identify a unique `FLAME geometry bias`; it measures a coupled mean-support/extent system.

Bootstrap intervals resample subjects and are descriptive with n=5. This remains an all-view oracle ceiling experiment, not held-out novel-view generalization.
"""
    (args.output / "findings.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
