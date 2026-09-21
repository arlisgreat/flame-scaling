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


CONDITIONS = ("hard", "r075", "free")
REGIONS = ("supported", "unsupported", "head")


def bootstrap_subjects(
    values: list[float], replicates: int = 10000, seed: int = 20260715
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    samples = array[indices].mean(axis=1)
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
    fold_rows: list[dict[str, object]] = []
    camera_rows: list[dict[str, object]] = []
    reports: dict[tuple[str, str, str], dict[str, object]] = {}
    for case in config["cases"]:
        subject = case["subject"]
        folds = sorted(path for path in (args.input / subject).glob("fold_*") if path.is_dir())
        if len(folds) != 3:
            raise RuntimeError(f"{subject}: expected three folds, found {len(folds)}")
        for fold_path in folds:
            for condition in CONDITIONS:
                report = json.loads((fold_path / condition / "metrics.json").read_text())
                reports[(subject, fold_path.name, condition)] = report
                fold_rows.append(
                    {
                        "subject": subject,
                        "sequence": case["sequence"],
                        "fold": fold_path.name,
                        "condition": condition,
                        "heldout_camera_count": len(report["heldout_camera_ids"]),
                        **{
                            f"{region}_train_psnr": report["evaluation"][region]["train"][
                                "masked_psnr"
                            ]
                            for region in REGIONS
                        },
                        **{
                            f"{region}_heldout_psnr": report["evaluation"][region]["heldout"][
                                "masked_psnr"
                            ]
                            for region in REGIONS
                        },
                    }
                )
                heldout_ids = set(report["heldout_camera_ids"])
                for region in REGIONS:
                    for item in report["evaluation"][region]["per_camera"]:
                        if item["camera_id"] in heldout_ids:
                            camera_rows.append(
                                {
                                    "subject": subject,
                                    "fold": fold_path.name,
                                    "condition": condition,
                                    "camera_id": item["camera_id"],
                                    "region": region,
                                    "psnr": item["masked_psnr"],
                                }
                            )

    for case in config["cases"]:
        subject = case["subject"]
        camera_ids = {
            str(row["camera_id"])
            for row in camera_rows
            if row["subject"] == subject
            and row["condition"] == "hard"
            and row["region"] == "head"
        }
        if len(camera_ids) != 13:
            raise RuntimeError(f"{subject}: held-out coverage has {len(camera_ids)} cameras")

    subject_rows: list[dict[str, object]] = []
    for case in config["cases"]:
        subject = case["subject"]
        for condition in CONDITIONS:
            row: dict[str, object] = {
                "subject": subject,
                "sequence": case["sequence"],
                "stress_region": case["stress_region"],
                "condition": condition,
            }
            for region in REGIONS:
                values = [
                    float(item["psnr"])
                    for item in camera_rows
                    if item["subject"] == subject
                    and item["condition"] == condition
                    and item["region"] == region
                ]
                row[f"{region}_heldout_psnr"] = float(np.mean(values))
                folds = [
                    item
                    for item in fold_rows
                    if item["subject"] == subject and item["condition"] == condition
                ]
                row[f"{region}_train_psnr"] = float(
                    np.average(
                        [float(item[f"{region}_train_psnr"]) for item in folds],
                        weights=[13 - int(item["heldout_camera_count"]) for item in folds],
                    )
                )
                row[f"{region}_generalization_gap_db"] = (
                    float(row[f"{region}_train_psnr"])
                    - float(row[f"{region}_heldout_psnr"])
                )
            subject_rows.append(row)

    lookup = {
        (str(row["subject"]), str(row["condition"])): row for row in subject_rows
    }
    effects: dict[str, object] = {}
    for region in REGIONS:
        metric = f"{region}_heldout_psnr"
        for name, left, right in (
            ("bounded_minus_hard", "r075", "hard"),
            ("bounded_minus_free", "r075", "free"),
            ("free_minus_hard", "free", "hard"),
        ):
            values = [
                float(lookup[(case["subject"], left)][metric])
                - float(lookup[(case["subject"], right)][metric])
                for case in config["cases"]
            ]
            effects[f"{region}_{name}"] = bootstrap_subjects(values)

    condition_summary = []
    for condition in CONDITIONS:
        selected = [row for row in subject_rows if row["condition"] == condition]
        condition_summary.append(
            {
                "condition": condition,
                **{
                    f"{region}_heldout_psnr": float(
                        np.mean([float(row[f"{region}_heldout_psnr"]) for row in selected])
                    )
                    for region in REGIONS
                },
                **{
                    f"{region}_generalization_gap_db": float(
                        np.mean(
                            [float(row[f"{region}_generalization_gap_db"]) for row in selected]
                        )
                    )
                    for region in REGIONS
                },
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (args.output / "per_fold.csv", fold_rows),
        (args.output / "per_camera.csv", camera_rows),
        (args.output / "per_subject.csv", subject_rows),
        (args.output / "condition_summary.csv", condition_summary),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "paired_effects.json").write_text(json.dumps(effects, indent=2) + "\n")

    subjects = [case["subject"] for case in config["cases"]]
    x = np.arange(len(subjects))
    styles = {
        "hard": ("#d62728", "Hard"),
        "r075": ("#ff7f0e", "75 mm"),
        "free": ("#1f77b4", "Free"),
    }
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.0))
    for axis, region in zip(axes[:3], REGIONS):
        for condition in CONDITIONS:
            color, label = styles[condition]
            axis.plot(
                x,
                [
                    float(lookup[(subject, condition)][f"{region}_heldout_psnr"])
                    for subject in subjects
                ],
                "o-",
                color=color,
                label=label,
            )
        axis.set_xticks(x, subjects)
        axis.set_xlabel("Subject")
        axis.set_ylabel("Held-out PSNR (dB)")
        axis.set_title(region.capitalize())
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    width = 0.24
    for offset, condition in zip((-width, 0.0, width), CONDITIONS):
        color, label = styles[condition]
        axes[3].bar(
            np.arange(len(REGIONS)) + offset,
            [
                next(
                    float(row[f"{region}_generalization_gap_db"])
                    for row in condition_summary
                    if row["condition"] == condition
                )
                for region in REGIONS
            ],
            width=width,
            color=color,
            label=label,
        )
    axes[3].set_xticks(np.arange(len(REGIONS)), REGIONS, rotation=20)
    axes[3].set_ylabel("Train−held-out PSNR (dB)")
    axes[3].set_title("Generalization gap")
    axes[3].grid(axis="y", alpha=0.25)
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle("Five-subject, three-fold angular cross-validation")
    fig.tight_layout()
    fig.savefig(args.output / "multicase_release_cv.png", dpi=200)
    plt.close(fig)

    dataset_root = Path(config["dataset_root"])
    fig, axes = plt.subplots(len(subjects), 4, figsize=(8.6, 10.0))
    columns = ("target", "hard", "r075", "free")
    labels = ("Target", "Hard", "75 mm", "Free")
    for row_index, case in enumerate(config["cases"]):
        subject = case["subject"]
        fold = "fold_a"
        report = reports[(subject, fold, "hard")]
        camera_id = report["heldout_camera_ids"][0]
        frame = load_multiview_frame(
            dataset_root / subject,
            case["sequence"],
            case["frame"],
            int(report["resolution"][0]),
            background_threshold=30,
        )
        index = frame["camera_ids"].index(camera_id)
        for column_index, condition in enumerate(columns):
            image = (
                frame["images"][index]
                if condition == "target"
                else read_rgb(args.input / subject / fold / condition / f"render_{camera_id}.png")
            )
            axes[row_index, column_index].imshow(np.clip(image, 0, 1))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(labels[column_index], fontsize=9)
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(subject, fontsize=9)
    fig.tight_layout(pad=0.25)
    fig.savefig(args.output / "qualitative_multicase_heldout.png", dpi=200)
    plt.close(fig)

    optimal_counts = {
        region: {
            condition: sum(
                max(
                    CONDITIONS,
                    key=lambda candidate: float(
                        lookup[(subject, candidate)][f"{region}_heldout_psnr"]
                    ),
                )
                == condition
                for subject in subjects
            )
            for condition in CONDITIONS
        }
        for region in REGIONS
    }
    findings = {
        "kind": "five_subject_three_fold_release_cross_validation",
        "design": {
            "subjects": subjects,
            "camera_count_per_subject": 13,
            "folds": 3,
            "conditions": {"hard": 0, "r075": 75, "free": "unbounded"},
            "gaussian_axis_cap_mm": 30,
            "steps": 4000,
        },
        "condition_summary": condition_summary,
        "effects": effects,
        "per_subject_optimal_counts": optimal_counts,
        "scope": "Subject bootstrap is descriptive with n=5; regions are FLAME-distance bands.",
    }
    (args.output / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")

    supported_bh = effects["supported_bounded_minus_hard"]
    supported_bf = effects["supported_bounded_minus_free"]
    unsupported_bf = effects["unsupported_bounded_minus_free"]
    unsupported_fh = effects["unsupported_free_minus_hard"]
    report = f"""# Five-subject cross-validation of bounded release

## Design

- Five NeRSemble NVS cases; all 13 cameras per subject are held out exactly once across three angularly interleaved folds.
- Externally selected conditions from the tongue pilot: hard, 75 mm bounded mean release, and free means; 30 mm Gaussian-axis cap and 4,000 steps throughout.

## Result

- Supported region, 75 mm minus hard: **{supported_bh['mean']:+.2f} dB**, subject bootstrap 95% CI [{supported_bh['ci95_low']:+.2f}, {supported_bh['ci95_high']:+.2f}].
- Supported region, 75 mm minus free: **{supported_bf['mean']:+.2f} dB** [{supported_bf['ci95_low']:+.2f}, {supported_bf['ci95_high']:+.2f}].
- Unsupported region, 75 mm minus free: **{unsupported_bf['mean']:+.2f} dB** [{unsupported_bf['ci95_low']:+.2f}, {unsupported_bf['ci95_high']:+.2f}].
- Unsupported region, free minus hard: **{unsupported_fh['mean']:+.2f} dB** [{unsupported_fh['ci95_low']:+.2f}, {unsupported_fh['ci95_high']:+.2f}].
- Per-subject optimal-condition counts: supported `{optimal_counts['supported']}`; unsupported `{optimal_counts['unsupported']}`.

## Interpretation

This tests whether the region-dependent bounded-release witness transfers beyond the selecting tongue case. Positive bounded-vs-free supported effects would support regularization; negative unsupported effects would preserve the representation-ceiling side. Regions here are geometric distance bands, not manual semantic masks, so the result cannot replace oral/hair-specific replication.

## Scope

The subject bootstrap is descriptive with n=5. These are per-instance optimizations with held-out cameras, not a cross-identity trainable scaling law.
"""
    (args.output / "findings.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
