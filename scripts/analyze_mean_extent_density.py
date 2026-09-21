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
DENSITIES = ((0.25, "d025"), (1.0, None), (4.0, "d400"))
REGIONS = ("supported", "unsupported", "head")


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
        "positive_subjects": int(np.sum(array > 0)),
        "per_subject": array.tolist(),
    }


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-input", type=Path, required=True)
    parser.add_argument("--unit-input", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("configs/oracle_cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.cases.read_text())
    rows: list[dict[str, object]] = []
    reports: dict[tuple[str, float, str], dict[str, object]] = {}
    run_directories: dict[tuple[str, float, str], Path] = {}
    for case in config["cases"]:
        subject = case["subject"]
        for density, density_directory in DENSITIES:
            root = args.unit_input if density_directory is None else args.density_input / density_directory
            for condition, mean_constraint, cap_mm in CONDITIONS:
                directory = root / subject / condition
                report = json.loads((directory / "metrics.json").read_text())
                model = np.load(directory / "model.npz")
                axis_max_mm = model["scales"].max(axis=1) * 1000.0
                reports[(subject, density, condition)] = report
                run_directories[(subject, density, condition)] = directory
                rows.append(
                    {
                        "subject": subject,
                        "sequence": case["sequence"],
                        "stress_region": case["stress_region"],
                        "density_multiplier": density,
                        "gaussian_count": report["gaussian_count"],
                        "condition": condition,
                        "mean_constraint": mean_constraint,
                        "scale_cap_mm": cap_mm,
                        "supported_psnr": report["evaluation"]["supported"]["masked_psnr"],
                        "unsupported_psnr": report["evaluation"]["unsupported"]["masked_psnr"],
                        "head_psnr": report["evaluation"]["head"]["masked_psnr"],
                        "supported_center_error_mm": report["geometry"][
                            "pcd_to_means_supported_mean_mm"
                        ],
                        "unsupported_center_error_mm": report["geometry"][
                            "pcd_to_means_unsupported_mean_mm"
                        ],
                        "axis_max_p90_mm": float(np.quantile(axis_max_mm, 0.90)),
                        "axis_fraction_at_cap": float(np.mean(axis_max_mm >= 0.999 * cap_mm)),
                        "total_trainable_parameters": report["total_trainable_parameters"],
                        "final_loss": report["history"][-1]["loss"],
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_case_condition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_key = {
        (str(row["subject"]), float(row["density_multiplier"]), str(row["condition"])): row
        for row in rows
    }
    effects: dict[str, object] = {}
    for region in REGIONS:
        metric = f"{region}_psnr"
        region_effects: dict[str, object] = {}
        for condition, _, _ in CONDITIONS:
            values = [
                float(by_key[(case["subject"], 4.0, condition)][metric])
                - float(by_key[(case["subject"], 0.25, condition)][metric])
                for case in config["cases"]
            ]
            region_effects[f"density_gain_d400_minus_d025_{condition}"] = bootstrap_mean(values)

        for density, _ in DENSITIES:
            for cap_name in ("003", "030"):
                values = [
                    float(by_key[(case["subject"], density, f"free_cap{cap_name}")][metric])
                    - float(by_key[(case["subject"], density, f"hard_cap{cap_name}")][metric])
                    for case in config["cases"]
                ]
                region_effects[f"release_gain_free_minus_hard_d{density:g}_cap{cap_name}"] = (
                    bootstrap_mean(values)
                )
            values = []
            for case in config["cases"]:
                subject = case["subject"]
                h3 = float(by_key[(subject, density, "hard_cap003")][metric])
                h30 = float(by_key[(subject, density, "hard_cap030")][metric])
                f3 = float(by_key[(subject, density, "free_cap003")][metric])
                f30 = float(by_key[(subject, density, "free_cap030")][metric])
                values.append((h30 - h3) - (f30 - f3))
            region_effects[f"mean_extent_interaction_d{density:g}"] = bootstrap_mean(values)

        for cap_name in ("003", "030"):
            values = []
            for case in config["cases"]:
                subject = case["subject"]
                hard_gain = (
                    float(by_key[(subject, 4.0, f"hard_cap{cap_name}")][metric])
                    - float(by_key[(subject, 0.25, f"hard_cap{cap_name}")][metric])
                )
                free_gain = (
                    float(by_key[(subject, 4.0, f"free_cap{cap_name}")][metric])
                    - float(by_key[(subject, 0.25, f"free_cap{cap_name}")][metric])
                )
                values.append(hard_gain - free_gain)
            region_effects[f"mean_density_interaction_hard_minus_free_cap{cap_name}"] = (
                bootstrap_mean(values)
            )

        for mean_constraint in ("hard", "free"):
            values = []
            for case in config["cases"]:
                subject = case["subject"]
                tight_gain = (
                    float(by_key[(subject, 4.0, f"{mean_constraint}_cap003")][metric])
                    - float(by_key[(subject, 0.25, f"{mean_constraint}_cap003")][metric])
                )
                wide_gain = (
                    float(by_key[(subject, 4.0, f"{mean_constraint}_cap030")][metric])
                    - float(by_key[(subject, 0.25, f"{mean_constraint}_cap030")][metric])
                )
                values.append(wide_gain - tight_gain)
            region_effects[f"density_extent_interaction_wide_minus_tight_{mean_constraint}"] = (
                bootstrap_mean(values)
            )

        three_way = []
        for case in config["cases"]:
            subject = case["subject"]

            def interaction(density: float) -> float:
                h3 = float(by_key[(subject, density, "hard_cap003")][metric])
                h30 = float(by_key[(subject, density, "hard_cap030")][metric])
                f3 = float(by_key[(subject, density, "free_cap003")][metric])
                f30 = float(by_key[(subject, density, "free_cap030")][metric])
                return (h30 - h3) - (f30 - f3)

            three_way.append(interaction(4.0) - interaction(0.25))
        region_effects["mean_extent_density_three_way_d400_minus_d025"] = bootstrap_mean(
            three_way
        )
        effects[region] = region_effects

    geometry_effects: dict[str, object] = {}
    for mean_constraint in ("hard", "free"):
        condition = f"{mean_constraint}_cap003"
        for region in ("supported", "unsupported"):
            metric = f"{region}_center_error_mm"
            values = [
                float(by_key[(case["subject"], 4.0, condition)][metric])
                - float(by_key[(case["subject"], 0.25, condition)][metric])
                for case in config["cases"]
            ]
            geometry_effects[f"{mean_constraint}_{region}_error_d400_minus_d025"] = (
                bootstrap_mean(values)
            )
            geometry_effects[f"{mean_constraint}_{region}_error_d400_absolute"] = bootstrap_mean(
                [
                    float(by_key[(case["subject"], 4.0, condition)][metric])
                    for case in config["cases"]
                ]
            )
    effects["geometry"] = geometry_effects
    (args.output / "paired_effects.json").write_text(json.dumps(effects, indent=2) + "\n")

    subjects = [case["subject"] for case in config["cases"]]
    density_values = np.array([density for density, _ in DENSITIES])
    styles = {
        "hard_cap003": ("Hard / 3 mm", "#d62728", "--"),
        "hard_cap030": ("Hard / 30 mm", "#d62728", "-"),
        "free_cap003": ("Free / 3 mm", "#1f77b4", "--"),
        "free_cap030": ("Free / 30 mm", "#1f77b4", "-"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2))
    for axis, region in zip(axes, REGIONS):
        metric = f"{region}_psnr"
        for condition, (label, color, linestyle) in styles.items():
            means = [
                np.mean([float(by_key[(subject, density, condition)][metric]) for subject in subjects])
                for density in density_values
            ]
            axis.plot(density_values, means, "o", color=color, ls=linestyle, label=label)
            for subject in subjects:
                values = [
                    float(by_key[(subject, density, condition)][metric])
                    for density in density_values
                ]
                axis.plot(density_values, values, color=color, ls=linestyle, alpha=0.12, lw=0.8)
        axis.set_xscale("log", base=4)
        axis.set_xticks(density_values, ["0.25×", "1×", "4×"])
        axis.set_xlabel("FLAME-surface Gaussian density")
        axis.set_ylabel("PSNR (dB)")
        axis.set_title(region.capitalize())
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "mean_extent_density_curves.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7))
    for axis, region in zip(axes, REGIONS):
        metric = f"{region}_psnr"
        matrix = np.zeros((3, 2), dtype=np.float64)
        for row_index, density in enumerate(density_values):
            for column_index, cap_name in enumerate(("003", "030")):
                matrix[row_index, column_index] = np.mean(
                    [
                        float(by_key[(subject, density, f"free_cap{cap_name}")][metric])
                        - float(by_key[(subject, density, f"hard_cap{cap_name}")][metric])
                        for subject in subjects
                    ]
                )
        limit = max(1.0, float(np.abs(matrix).max()))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        for row_index in range(3):
            for column_index in range(2):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:+.1f}", ha="center", va="center")
        axis.set_xticks([0, 1], ["3 mm", "30 mm"])
        axis.set_yticks([0, 1, 2], ["0.25×", "1×", "4×"])
        axis.set_xlabel("Covariance axis cap")
        axis.set_ylabel("Density")
        axis.set_title(f"{region}: free − hard")
        fig.colorbar(image, ax=axis, shrink=0.8, label="PSNR difference (dB)")
    fig.tight_layout()
    fig.savefig(args.output / "release_gap_density_extent_heatmaps.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(len(subjects), 6, figsize=(12.0, 9.8))
    columns = (
        ("Target", None, None),
        ("Hard/3, 0.25×", 0.25, "hard_cap003"),
        ("Hard/3, 4×", 4.0, "hard_cap003"),
        ("Hard/30, 0.25×", 0.25, "hard_cap030"),
        ("Hard/30, 4×", 4.0, "hard_cap030"),
        ("Free/3, 4×", 4.0, "free_cap003"),
    )
    dataset_root = Path(config["dataset_root"])
    for row_index, case in enumerate(config["cases"]):
        subject = case["subject"]
        reference = reports[(subject, 0.25, "hard_cap003")]
        frame = load_multiview_frame(
            dataset_root / subject,
            case["sequence"],
            case["frame"],
            int(reference["resolution"][0]),
            background_threshold=30,
        )
        preferred = "222200041"
        camera_id = preferred if preferred in frame["camera_ids"] else frame["camera_ids"][0]
        camera_index = frame["camera_ids"].index(camera_id)
        for column_index, (label, density, condition) in enumerate(columns):
            image = (
                frame["images"][camera_index]
                if condition is None
                else read_rgb(run_directories[(subject, float(density), condition)] / f"render_{camera_id}.png")
            )
            axes[row_index, column_index].imshow(np.clip(image, 0, 1))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(label, fontsize=8)
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(subject, fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.savefig(args.output / "qualitative_density_factorial.png", dpi=220)
    plt.close(fig)

    unsupported = effects["unsupported"]
    hard_tight = unsupported["density_gain_d400_minus_d025_hard_cap003"]
    hard_wide = unsupported["density_gain_d400_minus_d025_hard_cap030"]
    free_tight = unsupported["density_gain_d400_minus_d025_free_cap003"]
    three_way = unsupported["mean_extent_density_three_way_d400_minus_d025"]
    hard_density_extent = unsupported["density_extent_interaction_wide_minus_tight_hard"]
    hard_geometry_change = effects["geometry"]["hard_unsupported_error_d400_minus_d025"]
    hard_geometry_floor = effects["geometry"]["hard_unsupported_error_d400_absolute"]
    free_geometry_floor = effects["geometry"]["free_unsupported_error_d400_absolute"]
    report = f"""# `G_mu × G_Sigma × G_density` factorial

## Design

- Five landmark-anchored NeRSemble NVS cases.
- `mean={{hard,free}} × axis-cap={{3,30}} mm × density={{0.25×,1×,4×}}`.
- Within every density, hard/free have identical Gaussian counts, appearance parameters, optimizer, 4,000 steps, and all calibrated views.
- Low-density levels are nested FPS subsets; high density retains all FLAME vertices and adds deterministic area-sampled points on the same FLAME surface.

## Paired population effects

- Unsupported PSNR density gain (4× minus 0.25×), hard/3 mm: **{hard_tight['mean']:+.2f} dB** [{hard_tight['ci95_low']:+.2f}, {hard_tight['ci95_high']:+.2f}].
- Unsupported PSNR density gain, hard/30 mm: **{hard_wide['mean']:+.2f} dB** [{hard_wide['ci95_low']:+.2f}, {hard_wide['ci95_high']:+.2f}].
- Unsupported PSNR density gain, free/3 mm: **{free_tight['mean']:+.2f} dB** [{free_tight['ci95_low']:+.2f}, {free_tight['ci95_high']:+.2f}].
- Within hard means, density×extent interaction `(4×−0.25× at 30 mm) − (4×−0.25× at 3 mm)`: **{hard_density_extent['mean']:+.2f} dB** [{hard_density_extent['ci95_low']:+.2f}, {hard_density_extent['ci95_high']:+.2f}].
- Unsupported three-way interaction `(mean×extent at 4×) − (mean×extent at 0.25×)`: **{three_way['mean']:+.2f} dB** [{three_way['ci95_low']:+.2f}, {three_way['ci95_high']:+.2f}].
- Hard unsupported center-error change, 4× minus 0.25×: **{hard_geometry_change['mean']:+.2f} mm** [{hard_geometry_change['ci95_low']:+.2f}, {hard_geometry_change['ci95_high']:+.2f}].
- At 4×, absolute unsupported center error remains **{hard_geometry_floor['mean']:.2f} mm** [{hard_geometry_floor['ci95_low']:.2f}, {hard_geometry_floor['ci95_high']:.2f}] for hard versus **{free_geometry_floor['mean']:.2f} mm** [{free_geometry_floor['ci95_low']:.2f}, {free_geometry_floor['ci95_high']:.2f}] for free.

## Interpretation rule

Density is a third geometry-capacity channel. A large hard density gain with wide but not tight covariance is evidence of a cardinality-amplified covariance bypass, not evidence that the FLAME support surface became correct. A residual hard center-error floor at 4× is the actual support ceiling. Intervals resample five subjects and remain descriptive.
"""
    (args.output / "findings.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
