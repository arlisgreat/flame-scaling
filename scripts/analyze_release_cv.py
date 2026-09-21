#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from flame_oracle.io import load_multiview_frame
from flame_oracle.metrics import masked_image_metrics
from scripts.analyze_tongue_ladder import (
    build_oral_protrusion_masks,
    crop_around_mask,
    read_rgb,
)


CONDITIONS = [
    ("hard", 0.0),
    ("r010", 10.0),
    ("r025", 25.0),
    ("r050", 50.0),
    ("r075", 75.0),
    ("r100", 100.0),
    ("r150", 150.0),
    ("r200", 200.0),
    ("free", -1.0),
]


def mean_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def bootstrap_paired(
    values: np.ndarray, replicates: int = 10000, seed: int = 20260715
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = values[sample_indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def nondominated(rows: list[dict[str, object]]) -> list[str]:
    result = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            oral_better = float(other["oral_heldout_psnr"]) >= float(
                candidate["oral_heldout_psnr"]
            )
            unsupported_better = float(other["unsupported_heldout_psnr"]) >= float(
                candidate["unsupported_heldout_psnr"]
            )
            strictly_better = (
                float(other["oral_heldout_psnr"]) > float(candidate["oral_heldout_psnr"])
                or float(other["unsupported_heldout_psnr"])
                > float(candidate["unsupported_heldout_psnr"])
            )
            if oral_better and unsupported_better and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(str(candidate["condition"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--mask-height", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    folds = sorted(path for path in args.input.glob("fold_*") if path.is_dir())
    if len(folds) != 4:
        raise RuntimeError(f"expected four folds, found {len(folds)}")
    first_report = json.loads((folds[0] / "hard" / "metrics.json").read_text())
    height, width = first_report["resolution"]
    case = load_multiview_frame(
        args.subject_root, args.sequence, args.frame, height, background_threshold=30
    )
    high_case = load_multiview_frame(
        args.subject_root, args.sequence, args.frame, args.mask_height, background_threshold=30
    )
    high_masks, mask_report = build_oral_protrusion_masks(high_case)
    oral_masks = np.stack(
        [
            cv2.resize(
                mask[..., 0].astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            > 0
            for mask in high_masks
        ]
    )[..., None]
    camera_index = {camera_id: index for index, camera_id in enumerate(case["camera_ids"])}

    per_camera_rows: list[dict[str, object]] = []
    per_fold_rows: list[dict[str, object]] = []
    render_cache: dict[tuple[str, str, str], np.ndarray] = {}
    for fold_path in folds:
        fold = fold_path.name
        for condition, radius_mm in CONDITIONS:
            report = json.loads((fold_path / condition / "metrics.json").read_text())
            heldout_ids = set(report["heldout_camera_ids"])
            region_camera = {
                region: {
                    item["camera_id"]: item
                    for item in report["evaluation"][region]["per_camera"]
                }
                for region in ("supported", "unsupported", "head")
            }
            condition_rows = []
            for camera_id in case["camera_ids"]:
                index = camera_index[camera_id]
                render = read_rgb(fold_path / condition / f"render_{camera_id}.png")
                render_cache[(fold, condition, camera_id)] = render
                oral = masked_image_metrics(render, case["images"][index], oral_masks[index])
                row = {
                    "fold": fold,
                    "condition": condition,
                    "radius_mm": radius_mm,
                    "camera_id": camera_id,
                    "split": "heldout" if camera_id in heldout_ids else "train",
                    "oral_psnr": oral["masked_psnr"],
                    "oral_mae": oral["masked_mae"],
                    "supported_psnr": region_camera["supported"][camera_id]["masked_psnr"],
                    "unsupported_psnr": region_camera["unsupported"][camera_id]["masked_psnr"],
                    "head_psnr": region_camera["head"][camera_id]["masked_psnr"],
                }
                per_camera_rows.append(row)
                condition_rows.append(row)
            train = [row for row in condition_rows if row["split"] == "train"]
            heldout = [row for row in condition_rows if row["split"] == "heldout"]
            per_fold_rows.append(
                {
                    "fold": fold,
                    "condition": condition,
                    "radius_mm": radius_mm,
                    "oral_train_psnr": float(np.mean([row["oral_psnr"] for row in train])),
                    "oral_heldout_psnr": float(
                        np.mean([row["oral_psnr"] for row in heldout])
                    ),
                    "unsupported_train_psnr": report["evaluation"]["unsupported"]["train"][
                        "masked_psnr"
                    ],
                    "unsupported_heldout_psnr": report["evaluation"]["unsupported"][
                        "heldout"
                    ]["masked_psnr"],
                    "head_train_psnr": report["evaluation"]["head"]["train"][
                        "masked_psnr"
                    ],
                    "head_heldout_psnr": report["evaluation"]["head"]["heldout"][
                        "masked_psnr"
                    ],
                }
            )

    all_heldout = [row for row in per_camera_rows if row["split"] == "heldout"]
    coverage = {}
    for camera_id in case["camera_ids"]:
        coverage[camera_id] = sorted(
            {
                str(row["fold"])
                for row in all_heldout
                if row["camera_id"] == camera_id
            }
        )
    if any(len(value) != 1 for value in coverage.values()):
        raise RuntimeError(f"folds do not form exact one-time held-out coverage: {coverage}")

    aggregate_rows = []
    for condition, radius_mm in CONDITIONS:
        selected = [row for row in all_heldout if row["condition"] == condition]
        aggregate_rows.append(
            {
                "condition": condition,
                "radius_mm": radius_mm,
                **mean_rows(
                    selected,
                    ("oral_psnr", "supported_psnr", "unsupported_psnr", "head_psnr"),
                ),
            }
        )
    for row in aggregate_rows:
        row["oral_heldout_psnr"] = row.pop("oral_psnr")
        row["supported_heldout_psnr"] = row.pop("supported_psnr")
        row["unsupported_heldout_psnr"] = row.pop("unsupported_psnr")
        row["head_heldout_psnr"] = row.pop("head_psnr")

    args.output.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (args.output / "per_camera.csv", per_camera_rows),
        (args.output / "per_fold.csv", per_fold_rows),
        (args.output / "aggregate.csv", aggregate_rows),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    aggregate = {str(row["condition"]): row for row in aggregate_rows}
    best_oral = max(aggregate_rows, key=lambda row: float(row["oral_heldout_psnr"]))
    best_unsupported = max(
        aggregate_rows, key=lambda row: float(row["unsupported_heldout_psnr"])
    )
    pareto = nondominated(aggregate_rows)

    by_camera_condition = {
        (str(row["camera_id"]), str(row["condition"])): row for row in all_heldout
    }

    def paired(left: str, right: str, metric: str) -> dict[str, float]:
        values = np.asarray(
            [
                float(by_camera_condition[(camera_id, left)][metric])
                - float(by_camera_condition[(camera_id, right)][metric])
                for camera_id in case["camera_ids"]
            ]
        )
        return bootstrap_paired(values)

    effects = {
        "best_oral_minus_hard_oral": paired(
            str(best_oral["condition"]), "hard", "oral_psnr"
        ),
        "best_oral_minus_free_oral": paired(
            str(best_oral["condition"]), "free", "oral_psnr"
        ),
        "free_minus_best_oral_unsupported": paired(
            "free", str(best_oral["condition"]), "unsupported_psnr"
        ),
        "free_minus_hard_unsupported": paired("free", "hard", "unsupported_psnr"),
        "r010_minus_free_oral": paired("r010", "free", "oral_psnr"),
        "r010_minus_hard_oral": paired("r010", "hard", "oral_psnr"),
    }

    finite_conditions = [item for item in CONDITIONS if item[1] >= 0]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8))
    for axis, heldout_key, train_key, title in (
        (axes[0], "oral_heldout_psnr", "oral_train_psnr", "Oral"),
        (
            axes[1],
            "unsupported_heldout_psnr",
            "unsupported_train_psnr",
            "Unsupported",
        ),
    ):
        for fold_path in folds:
            selected = {
                str(row["condition"]): row
                for row in per_fold_rows
                if row["fold"] == fold_path.name
            }
            axis.plot(
                [radius for _, radius in finite_conditions],
                [float(selected[name][heldout_key]) for name, _ in finite_conditions],
                "o-",
                alpha=0.28,
                lw=1,
            )
        axis.plot(
            [radius for _, radius in finite_conditions],
            [float(aggregate[name][heldout_key]) for name, _ in finite_conditions],
            "o-",
            color="black",
            lw=2.2,
            label="out-of-fold mean",
        )
        axis.axhline(
            float(aggregate["free"][heldout_key]),
            ls="--",
            color="#1f77b4",
            label="free",
        )
        axis.set_title(f"{title} held-out")
        axis.set_xlabel("Mean release radius (mm)")
        axis.set_ylabel("PSNR (dB)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

        gap_axis = axes[2] if title == "Oral" else axes[3]
        fold_gap = []
        for condition, radius in finite_conditions:
            rows = [
                row for row in per_fold_rows if row["condition"] == condition
            ]
            fold_gap.append(
                float(np.mean([float(row[train_key]) - float(row[heldout_key]) for row in rows]))
            )
        gap_axis.plot(
            [radius for _, radius in finite_conditions], fold_gap, "o-", color="black", lw=2
        )
        free_rows = [row for row in per_fold_rows if row["condition"] == "free"]
        free_gap = float(
            np.mean([float(row[train_key]) - float(row[heldout_key]) for row in free_rows])
        )
        gap_axis.axhline(free_gap, ls="--", color="#1f77b4", label="free")
        gap_axis.set_title(f"{title} train−held-out")
        gap_axis.set_xlabel("Mean release radius (mm)")
        gap_axis.set_ylabel("PSNR gap (dB)")
        gap_axis.grid(alpha=0.25)
        gap_axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Four-fold angular cross-validation")
    fig.tight_layout()
    fig.savefig(args.output / "cross_validated_release_curves.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.8, 4.8))
    for row in aggregate_rows:
        axis.scatter(
            float(row["unsupported_heldout_psnr"]),
            float(row["oral_heldout_psnr"]),
            s=55,
            color="#2ca02c" if row["condition"] in pareto else "#8c8c8c",
        )
        axis.annotate(
            str(row["condition"]),
            (float(row["unsupported_heldout_psnr"]), float(row["oral_heldout_psnr"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Unsupported held-out PSNR (dB)")
    axis.set_ylabel("Oral held-out PSNR (dB)")
    axis.set_title("Region trade-off and non-dominated release doses")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "regional_pareto.png", dpi=200)
    plt.close(fig)

    display = ["target", "hard", "r010", "r025", "r050", "r100", "r200", "free"]
    labels = ["Target", "Hard", "10 mm", "25 mm", "50 mm", "100 mm", "200 mm", "Free"]
    fig, axes = plt.subplots(len(folds), len(display), figsize=(14.4, 7.4))
    for row_index, fold_path in enumerate(folds):
        report = json.loads((fold_path / "hard" / "metrics.json").read_text())
        camera_id = report["heldout_camera_ids"][0]
        index = camera_index[camera_id]
        mask = oral_masks[index, ..., 0]
        for column_index, condition in enumerate(display):
            image = (
                case["images"][index]
                if condition == "target"
                else render_cache[(fold_path.name, condition, camera_id)]
            )
            axes[row_index, column_index].imshow(crop_around_mask(image, mask))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(labels[column_index], fontsize=9)
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(fold_path.name, fontsize=8)
    fig.tight_layout(pad=0.35)
    fig.savefig(args.output / "qualitative_cross_validated_oral.png", dpi=200)
    plt.close(fig)

    best_oral_name = str(best_oral["condition"])
    findings = {
        "kind": "four_fold_angular_cross_validated_release_curve",
        "case": {
            "subject": args.subject_root.name,
            "sequence": args.sequence,
            "frame": args.frame,
            "camera_count": len(case["camera_ids"]),
            "fold_count": len(folds),
            "heldout_per_fold": 4,
            "steps": first_report["steps"],
            "resolution": first_report["resolution"],
        },
        "mask": mask_report,
        "coverage": coverage,
        "aggregate": aggregate_rows,
        "best_oral_condition": best_oral_name,
        "best_unsupported_condition": str(best_unsupported["condition"]),
        "pareto_conditions": pareto,
        "paired_camera_bootstrap": effects,
        "scope": (
            "One subject/frame with exact four-fold angular camera coverage; camera bootstrap is "
            "descriptive and does not substitute for subject-level replication."
        ),
    }
    (args.output / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")

    oral_hard = effects["best_oral_minus_hard_oral"]
    oral_free = effects["best_oral_minus_free_oral"]
    unsupported_tradeoff = effects["free_minus_best_oral_unsupported"]
    report = f"""# Four-fold angular cross-validation of mean release

## Result

- Every one of the 16 cameras is held out exactly once; each fold optimizes the other 12 cameras.
- The best mean release for oral held-out PSNR is **{best_oral_name}** at **{best_oral['oral_heldout_psnr']:.2f} dB**.
- It exceeds hard by **{oral_hard['mean']:+.2f} dB**, camera bootstrap 95% CI [{oral_hard['ci95_low']:+.2f}, {oral_hard['ci95_high']:+.2f}], and free by **{oral_free['mean']:+.2f} dB** [{oral_free['ci95_low']:+.2f}, {oral_free['ci95_high']:+.2f}].
- The broader unsupported region instead prefers **{best_unsupported['condition']}** at **{best_unsupported['unsupported_heldout_psnr']:.2f} dB**. Free exceeds the oral-optimal condition there by **{unsupported_tradeoff['mean']:+.2f} dB** [{unsupported_tradeoff['ci95_low']:+.2f}, {unsupported_tradeoff['ci95_high']:+.2f}].
- Non-dominated release doses across oral and unsupported held-out PSNR: `{', '.join(pareto)}`.

## Finding

The same representation and data budget has no globally optimal FLAME mean-release radius. Tight/bounded release regularizes the oral region, while unsupported outer-head content requires much freer support. This is an empirical region-conditioned bias-variance trade-off and a direct motivation for region-aware release; it is not yet evidence that a learned APR gate beats the oracle-selected fixed doses.

## Scope

This is one subject and one static frame. The camera-level bootstrap quantifies view variation only; subject/sequence replication and dynamic animation tests remain required.
"""
    (args.output / "findings.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
