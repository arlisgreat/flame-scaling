#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from flame_oracle.flame import load_flame_linear_assets
from flame_oracle.io import load_multiview_frame
from flame_oracle.metrics import masked_image_metrics
from scripts.analyze_tongue_ladder import (
    build_oral_protrusion_masks,
    crop_around_mask,
    read_rgb,
)


DEFAULT_ASSET_ROOT = Path(
    "/data1/workspace/airulan/3dv/external/LAM/model_zoo/human_parametric_models/flame_vhap"
)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / max(float(weights.sum()), 1e-8))


def scale_statistics(
    scales_mm: np.ndarray,
    opacities: np.ndarray,
    mask: np.ndarray,
    cap_mm: float,
) -> dict[str, float]:
    axis_max = scales_mm[mask].max(axis=1)
    weights = opacities[mask]
    return {
        "count": int(mask.sum()),
        "axis_max_mean_mm": float(axis_max.mean()),
        "axis_max_opacity_weighted_mean_mm": weighted_mean(axis_max, weights),
        "axis_max_p50_mm": float(np.quantile(axis_max, 0.50)),
        "axis_max_p90_mm": float(np.quantile(axis_max, 0.90)),
        "axis_max_p99_mm": float(np.quantile(axis_max, 0.99)),
        "axis_max_max_mm": float(axis_max.max()),
        "fraction_above_12mm": float(np.mean(axis_max > 12.001)),
        "fraction_at_cap": float(np.mean(axis_max >= 0.999 * cap_mm)),
    }


def gather_conditions(
    scale_runs: Path, baseline_runs: Path | None
) -> list[dict[str, object]]:
    conditions: list[dict[str, object]] = []
    pattern = re.compile(r"^(hard|r050|free)_cap(\d+)$")
    for metrics_path in sorted(scale_runs.glob("*/metrics.json")):
        match = pattern.match(metrics_path.parent.name)
        if not match:
            continue
        report = json.loads(metrics_path.read_text())
        conditions.append(
            {
                "family": match.group(1),
                "cap_mm": float(match.group(2)),
                "directory": metrics_path.parent,
                "report": report,
            }
        )
    existing = {(str(item["family"]), float(item["cap_mm"])) for item in conditions}
    for family in ("hard", "r050", "free"):
        metrics_path = baseline_runs / family / "metrics.json" if baseline_runs else None
        if metrics_path and metrics_path.exists() and (family, 30.0) not in existing:
            conditions.append(
                {
                    "family": family,
                    "cap_mm": 30.0,
                    "directory": metrics_path.parent,
                    "report": json.loads(metrics_path.read_text()),
                }
            )
    return sorted(conditions, key=lambda item: (str(item["family"]), float(item["cap_mm"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-runs", type=Path, required=True)
    parser.add_argument("--baseline-runs", type=Path)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--mask-height", type=int, default=256)
    parser.add_argument("--mouth-neighborhood-mm", type=float, default=45.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conditions = gather_conditions(args.scale_runs, args.baseline_runs)
    if len(conditions) != 12:
        raise RuntimeError(f"expected 12 completed conditions, found {len(conditions)}")

    fit = np.load(args.fit)
    base = fit["vertices"].astype(np.float32)
    assets = load_flame_linear_assets(
        args.asset_root / "flame2023.pkl", args.asset_root / "FLAME_masks.pkl"
    )
    lips = np.asarray(assets["masks"]["lips"], dtype=np.int64)
    distance_to_lips = cKDTree(base[lips]).query(base, workers=-1)[0]
    mouth_neighborhood = distance_to_lips <= args.mouth_neighborhood_mm / 1000.0

    resolution_height, resolution_width = conditions[0]["report"]["resolution"]
    case = load_multiview_frame(
        args.subject_root,
        args.sequence,
        args.frame,
        resolution_height,
        background_threshold=30,
    )
    high_case = load_multiview_frame(
        args.subject_root,
        args.sequence,
        args.frame,
        args.mask_height,
        background_threshold=30,
    )
    high_masks, mask_report = build_oral_protrusion_masks(high_case)
    oral_masks = np.stack(
        [
            cv2.resize(
                mask[..., 0].astype(np.uint8),
                (resolution_width, resolution_height),
                interpolation=cv2.INTER_AREA,
            )
            > 0
            for mask in high_masks
        ]
    )[..., None]

    rows: list[dict[str, object]] = []
    render_cache: dict[tuple[str, float, str], np.ndarray] = {}
    detailed_scales: dict[str, object] = {}
    for condition in conditions:
        family = str(condition["family"])
        cap_mm = float(condition["cap_mm"])
        directory = Path(condition["directory"])
        report = condition["report"]
        model = np.load(directory / "model.npz")
        scales_mm = model["scales"].astype(np.float64) * 1000.0
        opacities = model["opacities"].astype(np.float64)
        offsets_mm = np.linalg.norm(model["means"] - base, axis=1) * 1000.0
        all_stats = scale_statistics(
            scales_mm, opacities, np.ones(len(base), dtype=bool), cap_mm
        )
        mouth_stats = scale_statistics(
            scales_mm, opacities, mouth_neighborhood, cap_mm
        )
        key = f"{family}_cap{int(cap_mm):03d}"
        detailed_scales[key] = {
            "all": all_stats,
            "mouth_neighborhood": mouth_stats,
        }

        per_camera = []
        for camera_index, camera_id in enumerate(case["camera_ids"]):
            render = read_rgb(directory / f"render_{camera_id}.png")
            render_cache[(family, cap_mm, camera_id)] = render
            per_camera.append(
                {
                    "camera_id": camera_id,
                    **masked_image_metrics(
                        render, case["images"][camera_index], oral_masks[camera_index]
                    ),
                }
            )
        oral_psnr = float(np.mean([item["masked_psnr"] for item in per_camera]))
        oral_mae = float(np.mean([item["masked_mae"] for item in per_camera]))
        train_ids = set(report.get("train_camera_ids", report["camera_ids"]))
        heldout_ids = set(report.get("heldout_camera_ids", []))
        train_oral = [item for item in per_camera if item["camera_id"] in train_ids]
        heldout_oral = [item for item in per_camera if item["camera_id"] in heldout_ids]
        evaluation = report["evaluation"]
        rows.append(
            {
                "family": family,
                "radius_mm": float(report["radius_mm"]),
                "scale_cap_mm": cap_mm,
                "oral_protrusion_psnr": oral_psnr,
                "oral_protrusion_mae": oral_mae,
                "oral_train_psnr": float(np.mean([item["masked_psnr"] for item in train_oral])),
                "oral_heldout_psnr": (
                    float(np.mean([item["masked_psnr"] for item in heldout_oral]))
                    if heldout_oral
                    else math.nan
                ),
                "unsupported_psnr": float(evaluation["unsupported"]["masked_psnr"]),
                "unsupported_train_psnr": float(
                    evaluation["unsupported"].get("train", evaluation["unsupported"])["masked_psnr"]
                ),
                "unsupported_heldout_psnr": (
                    float(evaluation["unsupported"]["heldout"]["masked_psnr"])
                    if evaluation["unsupported"].get("heldout")
                    else math.nan
                ),
                "head_psnr": float(evaluation["head"]["masked_psnr"]),
                "head_train_psnr": float(
                    evaluation["head"].get("train", evaluation["head"])["masked_psnr"]
                ),
                "head_heldout_psnr": (
                    float(evaluation["head"]["heldout"]["masked_psnr"])
                    if evaluation["head"].get("heldout")
                    else math.nan
                ),
                "unsupported_center_error_mm": float(
                    report["geometry"]["pcd_to_means_unsupported_mean_mm"]
                ),
                "mean_offset_mm": float(offsets_mm.mean()),
                "mouth_axis_max_p90_mm": mouth_stats["axis_max_p90_mm"],
                "mouth_fraction_at_cap": mouth_stats["fraction_at_cap"],
                "all_axis_max_p90_mm": all_stats["axis_max_p90_mm"],
                "all_fraction_at_cap": all_stats["fraction_at_cap"],
            }
        )

    rows.sort(key=lambda row: (str(row["family"]), float(row["scale_cap_mm"])))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_condition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    family_labels = {"hard": "Hard means", "r050": "50 mm release", "free": "Free means"}
    family_colors = {"hard": "#d62728", "r050": "#ff7f0e", "free": "#1f77b4"}
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.7))
    for family in ("hard", "r050", "free"):
        selected = [row for row in rows if row["family"] == family]
        x = [float(row["scale_cap_mm"]) for row in selected]
        for axis, key in zip(
            axes,
            (
                "oral_protrusion_psnr",
                "unsupported_psnr",
                "unsupported_center_error_mm",
                "mouth_axis_max_p90_mm",
            ),
        ):
            axis.plot(
                x,
                [float(row[key]) for row in selected],
                "o-",
                lw=2,
                label=family_labels[family],
                color=family_colors[family],
            )
    axes[0].set_ylabel("Oral-protrusion PSNR (dB)")
    axes[1].set_ylabel("Unsupported PSNR (dB)")
    axes[2].set_ylabel("Unsupported center error (mm)")
    axes[3].set_ylabel("Mouth max-axis p90 (mm)")
    for axis in axes:
        axis.set_xlabel("Gaussian axis cap (mm)")
        axis.set_xticks([3, 6, 12, 30])
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Mean-position release × Gaussian-extent capacity")
    fig.tight_layout()
    fig.savefig(args.output / "mean_extent_interaction.png", dpi=200)
    plt.close(fig)

    has_heldout = any(not math.isnan(float(row["oral_heldout_psnr"])) for row in rows)
    if has_heldout:
        fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.7))
        for family in ("hard", "r050", "free"):
            selected = [row for row in rows if row["family"] == family]
            x = [float(row["scale_cap_mm"]) for row in selected]
            for axis, key in zip(
                axes,
                (
                    "oral_train_psnr",
                    "oral_heldout_psnr",
                    "unsupported_train_psnr",
                    "unsupported_heldout_psnr",
                ),
            ):
                axis.plot(
                    x,
                    [float(row[key]) for row in selected],
                    "o-",
                    lw=2,
                    label=family_labels[family],
                    color=family_colors[family],
                )
        for axis, title in zip(
            axes,
            ("Oral train", "Oral held-out", "Unsupported train", "Unsupported held-out"),
        ):
            axis.set_title(title)
            axis.set_xlabel("Gaussian axis cap (mm)")
            axis.set_ylabel("PSNR (dB)")
            axis.set_xticks([3, 6, 12, 30])
            axis.grid(alpha=0.25)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Train-view versus held-out-view interaction")
        fig.tight_layout()
        fig.savefig(args.output / "heldout_interaction.png", dpi=200)
        plt.close(fig)

    display = [
        ("target", math.nan),
        ("hard", 3.0),
        ("hard", 6.0),
        ("hard", 12.0),
        ("hard", 30.0),
        ("free", 3.0),
        ("free", 6.0),
        ("free", 12.0),
        ("free", 30.0),
    ]
    labels = ["Target", "Hard/3", "Hard/6", "Hard/12", "Hard/30", "Free/3", "Free/6", "Free/12", "Free/30"]
    camera_indices = [0, 4, 15]
    fig, axes = plt.subplots(len(camera_indices), len(display), figsize=(16.2, 5.6))
    for row_index, camera_index in enumerate(camera_indices):
        camera_id = case["camera_ids"][camera_index]
        mask = oral_masks[camera_index, ..., 0]
        for column_index, (family, cap_mm) in enumerate(display):
            image = (
                case["images"][camera_index]
                if family == "target"
                else render_cache[(family, cap_mm, camera_id)]
            )
            axes[row_index, column_index].imshow(crop_around_mask(image, mask))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(labels[column_index], fontsize=9)
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(camera_id, fontsize=8)
    fig.tight_layout(pad=0.35)
    fig.savefig(args.output / "qualitative_mean_extent_grid.png", dpi=200)
    plt.close(fig)

    if has_heldout:
        heldout_ids = set(conditions[0]["report"].get("heldout_camera_ids", []))
        heldout_camera_indices = [
            index for index, camera_id in enumerate(case["camera_ids"])
            if camera_id in heldout_ids
        ][:3]
        fig, axes = plt.subplots(
            len(heldout_camera_indices), len(display), figsize=(16.2, 5.6)
        )
        axes = np.atleast_2d(axes)
        for row_index, camera_index in enumerate(heldout_camera_indices):
            camera_id = case["camera_ids"][camera_index]
            mask = oral_masks[camera_index, ..., 0]
            for column_index, (family, cap_mm) in enumerate(display):
                image = (
                    case["images"][camera_index]
                    if family == "target"
                    else render_cache[(family, cap_mm, camera_id)]
                )
                axes[row_index, column_index].imshow(crop_around_mask(image, mask))
                axes[row_index, column_index].axis("off")
                if row_index == 0:
                    axes[row_index, column_index].set_title(labels[column_index], fontsize=9)
                if column_index == 0:
                    axes[row_index, column_index].set_ylabel(camera_id, fontsize=8)
        fig.tight_layout(pad=0.35)
        fig.savefig(args.output / "qualitative_heldout_mean_extent_grid.png", dpi=200)
        plt.close(fig)

    lookup = {(str(row["family"]), float(row["scale_cap_mm"])): row for row in rows}
    contrasts: dict[str, float] = {}
    for cap in (3.0, 6.0, 12.0, 30.0):
        hard = lookup[("hard", cap)]
        free = lookup[("free", cap)]
        contrasts[f"oral_free_minus_hard_cap{int(cap):03d}_db"] = float(
            free["oral_protrusion_psnr"] - hard["oral_protrusion_psnr"]
        )
        contrasts[f"unsupported_free_minus_hard_cap{int(cap):03d}_db"] = float(
            free["unsupported_psnr"] - hard["unsupported_psnr"]
        )
        if has_heldout:
            for metric in (
                "oral_train_psnr",
                "oral_heldout_psnr",
                "unsupported_train_psnr",
                "unsupported_heldout_psnr",
            ):
                contrasts[f"{metric}_free_minus_hard_cap{int(cap):03d}_db"] = float(
                    free[metric] - hard[metric]
                )
            r050 = lookup[("r050", cap)]
            for metric in ("oral_heldout_psnr", "unsupported_heldout_psnr"):
                contrasts[f"{metric}_r050_minus_free_cap{int(cap):03d}_db"] = float(
                    r050[metric] - free[metric]
                )
    hard_extent_gain = float(
        lookup[("hard", 30.0)]["oral_protrusion_psnr"]
        - lookup[("hard", 3.0)]["oral_protrusion_psnr"]
    )
    free_extent_gain = float(
        lookup[("free", 30.0)]["oral_protrusion_psnr"]
        - lookup[("free", 3.0)]["oral_protrusion_psnr"]
    )
    contrasts["oral_hard_cap030_minus_cap003_db"] = hard_extent_gain
    contrasts["oral_free_cap030_minus_cap003_db"] = free_extent_gain
    contrasts["oral_difference_in_extent_gains_db"] = hard_extent_gain - free_extent_gain

    findings = {
        "kind": "mean_position_by_gaussian_extent_interaction",
        "case": {
            "subject": args.subject_root.name,
            "sequence": args.sequence,
            "frame": args.frame,
            "views": len(case["camera_ids"]),
            "steps": conditions[0]["report"]["steps"],
            "resolution": conditions[0]["report"]["resolution"],
            "train_camera_ids": conditions[0]["report"].get(
                "train_camera_ids", conditions[0]["report"]["camera_ids"]
            ),
            "heldout_camera_ids": conditions[0]["report"].get("heldout_camera_ids", []),
        },
        "design": {
            "mean_release_mm": {"hard": 0, "r050": 50, "free": "unbounded"},
            "gaussian_axis_cap_mm": [3, 6, 12, 30],
            "soft_scale_penalty": "unchanged: quadratic above 12 mm, weight 100",
            "mouth_neighborhood_mm": args.mouth_neighborhood_mm,
            "mouth_vertex_count": int(mouth_neighborhood.sum()),
            "oral_mask": mask_report,
        },
        "conditions": rows,
        "scale_statistics": detailed_scales,
        "contrasts": contrasts,
        "scope": (
            "Static 12-train-camera/4-held-out-camera interaction test."
            if has_heldout
            else "Static all-view oracle interaction test."
        )
        + " It separates Gaussian mean support from Gaussian extent capacity.",
    }
    (args.output / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")

    hard3 = lookup[("hard", 3.0)]
    hard30 = lookup[("hard", 30.0)]
    free3 = lookup[("free", 3.0)]
    free30 = lookup[("free", 30.0)]
    heldout_text = ""
    if has_heldout:
        r050_30 = lookup[("r050", 30.0)]
        heldout_text = f"""
## Held-out cameras

- At a 3 mm extent cap, free-minus-hard oral PSNR is **{free3['oral_train_psnr'] - hard3['oral_train_psnr']:+.2f} dB** on optimized cameras and **{free3['oral_heldout_psnr'] - hard3['oral_heldout_psnr']:+.2f} dB** on held-out cameras.
- At 30 mm, the corresponding gaps are **{free30['oral_train_psnr'] - hard30['oral_train_psnr']:+.2f} dB** and **{free30['oral_heldout_psnr'] - hard30['oral_heldout_psnr']:+.2f} dB**.
- At 30 mm extent, 50 mm mean release is **{r050_30['oral_heldout_psnr'] - free30['oral_heldout_psnr']:+.2f} dB** versus free on held-out oral pixels, but **{r050_30['unsupported_heldout_psnr'] - free30['unsupported_heldout_psnr']:+.2f} dB** on the broader unsupported region. The region-specific optimum already differs in this case.
"""
    text = f"""# Gaussian extent as a hidden geometry-release channel

Case: subject `{args.subject_root.name}`, `{args.sequence}`, frame `{args.frame}`, 16 calibrated views; all conditions use 5,023 Gaussians and 4,000 optimization steps.

## Factorial result

- With a 3 mm axis cap, free means beat hard means in the oral-protrusion ROI by **{free3['oral_protrusion_psnr'] - hard3['oral_protrusion_psnr']:+.2f} dB**.
- With the original 30 mm cap, that same gap is only **{free30['oral_protrusion_psnr'] - hard30['oral_protrusion_psnr']:+.2f} dB**.
- Raising only the scale cap from 3 to 30 mm changes hard-mean oral PSNR by **{hard_extent_gain:+.2f} dB**, versus **{free_extent_gain:+.2f} dB** for free means. The difference in extent gains is **{hard_extent_gain - free_extent_gain:+.2f} dB**.
- Hard-mean unsupported center error remains {hard3['unsupported_center_error_mm']:.2f}–{hard30['unsupported_center_error_mm']:.2f} mm across scale caps: covariance can improve images without correcting center geometry.
- In the original hard/30 condition, the 90th percentile of the maximum axis within 45 mm of the fitted lips is **{hard30['mouth_axis_max_p90_mm']:.2f} mm**.

## Finding

`FLAME geometry bias` is not one knob in Gaussian avatars. Mean anchoring and covariance extent are separate geometry channels. A model with hard FLAME means can bypass part of the apparent representation ceiling by stretching splats, producing plausible RGB while retaining wrong center geometry. Any FLAME-bias scaling law that varies only query/mean locations is confounded unless Gaussian extent is controlled and geometry-sensitive metrics are reported.

{heldout_text}

## Scope

This is a single-frame mechanism test. {'Twelve cameras are optimized and four angularly interleaved cameras are held out.' if has_heldout else 'All cameras are optimized and evaluated.'} It does not establish a population scaling law.
"""
    (args.output / "findings.md").write_text(text)
    print(json.dumps({"output": str(args.output), "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
