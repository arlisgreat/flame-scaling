#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from flame_oracle.io import load_multiview_frame, save_montage
from flame_oracle.metrics import masked_image_metrics


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read render: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def build_oral_protrusion_masks(case: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    """Build a reviewed oral-protrusion polygon for the extended-tongue frame.

    MediaPipe's mouth corners remain stable here, while its nominal chin
    landmark is consistently attracted to the protruding tongue tip. The
    quadrilateral spanning those three anchors covers the visible tongue and
    a small part of the mouth cavity, without leaking onto the neck.
    """
    masks = []
    diagnostics = []
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as detector:
        for camera_id, image_float, alpha in zip(
            case["camera_ids"], case["images"], case["alpha"][..., 0]
        ):
            image = np.clip(image_float * 255.0, 0, 255).astype(np.uint8)
            result = detector.process(image)
            if not result.multi_face_landmarks:
                raise RuntimeError(f"MediaPipe failed on camera {camera_id}")
            landmarks = result.multi_face_landmarks[0].landmark
            image_size = np.asarray([image.shape[1], image.shape[0]], dtype=np.float32)
            left = np.asarray([landmarks[61].x, landmarks[61].y]) * image_size
            right = np.asarray([landmarks[291].x, landmarks[291].y]) * image_size
            tip = np.asarray([landmarks[152].x, landmarks[152].y]) * image_size
            width_vector = right - left
            polygon = np.stack(
                [left, right, tip + 0.28 * width_vector, tip - 0.28 * width_vector]
            ).round().astype(np.int32)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, polygon, 1)
            clean = (mask > 0) & (alpha > 0.5)
            masks.append(clean)
            diagnostics.append(
                {
                    "camera_id": camera_id,
                    "oral_protrusion_pixels": int(clean.sum()),
                    "mouth_left_xy": left.tolist(),
                    "mouth_right_xy": right.tolist(),
                    "tip_proxy_xy": tip.tolist(),
                }
            )
    return np.stack(masks)[..., None], {
        "method": "mediapipe_mouth_corners_plus_tongue_attracted_chin_landmark_polygon",
        "status": "exploratory_reviewed_roi_not_semantic_ground_truth",
        "polygon_tip_half_width_fraction": 0.28,
        "excluded_from_scaffold_fit": ["chin_landmark_152", "lower_lip_landmark_14"],
        "per_camera": diagnostics,
    }


def crop_around_mask(image: np.ndarray, mask: np.ndarray, size: int = 64) -> np.ndarray:
    height, width = image.shape[:2]
    y, x = np.where(mask)
    if len(x):
        center_x = int(round(float(x.mean())))
        center_y = int(round(float(y.mean())))
    else:
        center_x, center_y = width // 2, int(height * 0.62)
    half = size // 2
    x0 = max(0, min(width - size, center_x - half))
    y0 = max(0, min(height - size, center_y - half))
    crop = image[y0 : y0 + size, x0 : x0 + size]
    return np.clip(cv2.resize(crop, (128, 128), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--mask-height", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    high_case = load_multiview_frame(
        args.subject_root, args.sequence, args.frame, args.mask_height, background_threshold=30
    )
    high_masks, mask_report = build_oral_protrusion_masks(high_case)

    reports = []
    for metrics_path in sorted(args.runs.glob("*/metrics.json")):
        report = json.loads(metrics_path.read_text())
        report["directory"] = metrics_path.parent.name
        reports.append(report)
    if not reports:
        raise RuntimeError(f"no run reports under {args.runs}")
    resolution_height, resolution_width = reports[0]["resolution"]
    case = load_multiview_frame(
        args.subject_root, args.sequence, args.frame, resolution_height, background_threshold=30
    )
    oral_masks = np.stack(
        [
            cv2.resize(mask[..., 0].astype(np.uint8), (resolution_width, resolution_height), interpolation=cv2.INTER_AREA)
            > 0
            for mask in high_masks
        ]
    )[..., None]

    overlay_images = []
    for image, mask in zip(high_case["images"], high_masks[..., 0]):
        overlay = image.copy()
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([0.05, 1.0, 0.85])
        overlay_images.append(overlay)
    args.output.mkdir(parents=True, exist_ok=True)
    save_montage(args.output / "oral_protrusion_mask_audit.png", overlay_images, columns=4)

    rows = []
    render_cache: dict[tuple[str, str], np.ndarray] = {}
    for report in reports:
        per_camera = []
        for index, camera_id in enumerate(case["camera_ids"]):
            render = read_rgb(args.runs / report["directory"] / f"render_{camera_id}.png")
            render_cache[(report["directory"], camera_id)] = render
            per_camera.append(
                {
                    "camera_id": camera_id,
                    **masked_image_metrics(render, case["images"][index], oral_masks[index]),
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
                "directory": report["directory"],
                "radius_mm": float(report["radius_mm"]),
                "oral_protrusion_psnr": oral_psnr,
                "oral_protrusion_mae": oral_mae,
                "oral_train_psnr": float(np.mean([item["masked_psnr"] for item in train_oral])),
                "oral_heldout_psnr": (
                    float(np.mean([item["masked_psnr"] for item in heldout_oral]))
                    if heldout_oral
                    else math.nan
                ),
                "supported_psnr": float(evaluation["supported"]["masked_psnr"]),
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
                "unsupported_geometry_mm": float(
                    report["geometry"]["pcd_to_means_unsupported_mean_mm"]
                ),
                "mean_offset_mm": float(report["history"][-1]["mean_offset_mm"]),
            }
        )
    rows.sort(key=lambda row: (math.inf if row["radius_mm"] < 0 else row["radius_mm"]))

    with (args.output / "per_condition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    finite = [row for row in rows if row["radius_mm"] >= 0]
    free = next(row for row in rows if row["radius_mm"] < 0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for axis, key, ylabel in (
        (axes[0], "supported_psnr", "Supported PSNR (dB)"),
        (axes[1], "unsupported_psnr", "Unsupported PSNR (dB)"),
        (axes[2], "oral_protrusion_psnr", "Oral-protrusion PSNR (dB)"),
    ):
        axis.plot([row["radius_mm"] for row in finite], [row[key] for row in finite], "o-", lw=2)
        axis.axhline(free[key], ls="--", color="black", lw=1.2, label=f"free={free[key]:.2f}")
        axis.set_xlabel("Maximum release radius (mm)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Subject 017, EXP-6-tongue-1 frame 182, 16-view oracle fit")
    fig.tight_layout()
    fig.savefig(args.output / "release_curve.png", dpi=180)
    plt.close(fig)

    has_heldout = any(not math.isnan(float(row["oral_heldout_psnr"])) for row in rows)
    if has_heldout:
        fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8))
        for axis, key, title in (
            (axes[0], "oral_train_psnr", "Oral train"),
            (axes[1], "oral_heldout_psnr", "Oral held-out"),
            (axes[2], "unsupported_train_psnr", "Unsupported train"),
            (axes[3], "unsupported_heldout_psnr", "Unsupported held-out"),
        ):
            axis.plot(
                [row["radius_mm"] for row in finite],
                [row[key] for row in finite],
                "o-",
                lw=2,
            )
            axis.axhline(free[key], ls="--", color="black", lw=1.2, label=f"free={free[key]:.2f}")
            axis.set_xlabel("Maximum mean release (mm)")
            axis.set_ylabel("PSNR (dB)")
            axis.set_title(title)
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
        fig.suptitle("Region-specific held-out release curves")
        fig.tight_layout()
        fig.savefig(args.output / "heldout_release_curve.png", dpi=200)
        plt.close(fig)

    display_conditions = ["target", "hard", "r010", "r025", "r050", "r100", "r200", "free"]
    labels = ["Target", "Hard", "10 mm", "25 mm", "50 mm", "100 mm", "200 mm", "Free"]
    camera_indices = [0, 4, 15]
    fig, axes = plt.subplots(len(camera_indices), len(display_conditions), figsize=(14.4, 5.6))
    for row_index, camera_index in enumerate(camera_indices):
        camera_id = case["camera_ids"][camera_index]
        mask = oral_masks[camera_index, ..., 0]
        for column_index, condition in enumerate(display_conditions):
            image = (
                case["images"][camera_index]
                if condition == "target"
                else render_cache[(condition, camera_id)]
            )
            axes[row_index, column_index].imshow(crop_around_mask(image, mask))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(labels[column_index], fontsize=9)
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(camera_id, fontsize=8)
    fig.tight_layout(pad=0.35)
    fig.savefig(args.output / "qualitative_tongue_crops.png", dpi=200)
    plt.close(fig)

    if has_heldout:
        heldout_ids = set(reports[0].get("heldout_camera_ids", []))
        heldout_indices = [
            index for index, camera_id in enumerate(case["camera_ids"])
            if camera_id in heldout_ids
        ][:3]
        fig, axes = plt.subplots(len(heldout_indices), len(display_conditions), figsize=(14.4, 5.6))
        axes = np.atleast_2d(axes)
        for row_index, camera_index in enumerate(heldout_indices):
            camera_id = case["camera_ids"][camera_index]
            mask = oral_masks[camera_index, ..., 0]
            for column_index, condition in enumerate(display_conditions):
                image = (
                    case["images"][camera_index]
                    if condition == "target"
                    else render_cache[(condition, camera_id)]
                )
                axes[row_index, column_index].imshow(crop_around_mask(image, mask))
                axes[row_index, column_index].axis("off")
                if row_index == 0:
                    axes[row_index, column_index].set_title(labels[column_index], fontsize=9)
                if column_index == 0:
                    axes[row_index, column_index].set_ylabel(camera_id, fontsize=8)
        fig.tight_layout(pad=0.35)
        fig.savefig(args.output / "qualitative_heldout_tongue_crops.png", dpi=200)
        plt.close(fig)

    by_directory = {row["directory"]: row for row in rows}
    hard = by_directory["hard"]
    r100 = by_directory["r100"]
    r200 = by_directory["r200"]
    findings = {
        "kind": "single_subject_real_tongue_release_ladder",
        "case": {
            "subject": args.subject_root.name,
            "sequence": args.sequence,
            "frame": args.frame,
            "views": len(case["camera_ids"]),
            "steps": reports[0]["steps"],
            "resolution": reports[0]["resolution"],
            "train_camera_ids": reports[0].get("train_camera_ids", reports[0]["camera_ids"]),
            "heldout_camera_ids": reports[0].get("heldout_camera_ids", []),
        },
        "mask": mask_report,
        "conditions": rows,
        "contrasts_db": {
            "unsupported_free_minus_hard": free["unsupported_psnr"] - hard["unsupported_psnr"],
            "oral_protrusion_free_minus_hard": free["oral_protrusion_psnr"] - hard["oral_protrusion_psnr"],
            "unsupported_r100_minus_r050": r100["unsupported_psnr"] - by_directory["r050"]["unsupported_psnr"],
            "unsupported_free_minus_r200": free["unsupported_psnr"] - r200["unsupported_psnr"],
            "oral_protrusion_free_minus_r200": free["oral_protrusion_psnr"] - r200["oral_protrusion_psnr"],
        },
        "scope": (
            "Static 12-train-camera/4-held-out-camera optimization."
            if has_heldout
            else "Static all-view oracle optimization."
        ),
    }
    if has_heldout:
        findings["regional_optima"] = {
            "oral_heldout": max(rows, key=lambda row: row["oral_heldout_psnr"])["directory"],
            "unsupported_heldout": max(
                rows, key=lambda row: row["unsupported_heldout_psnr"]
            )["directory"],
        }
    (args.output / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")
    text = f"""# Real-tongue geometry release pilot

Case: subject `{args.subject_root.name}`, `{args.sequence}`, frame `{args.frame}`, 16 calibrated views, equal 4,000-step oracle fits.

## Result

- Unsupported-region PSNR improves from **{hard['unsupported_psnr']:.2f} dB** (hard) to **{free['unsupported_psnr']:.2f} dB** (free), a **{free['unsupported_psnr'] - hard['unsupported_psnr']:+.2f} dB** gap.
- The curve has a broad knee: 50 mm reaches {by_directory['r050']['unsupported_psnr']:.2f} dB, 75 mm {by_directory['r075']['unsupported_psnr']:.2f} dB, and 100 mm {r100['unsupported_psnr']:.2f} dB.
- A 200 mm release is close to free on this frame: unsupported gap **{free['unsupported_psnr'] - r200['unsupported_psnr']:+.2f} dB**; oral-protrusion gap **{free['oral_protrusion_psnr'] - r200['oral_protrusion_psnr']:+.2f} dB**.
- The exploratory oral-protrusion contrast is **{free['oral_protrusion_psnr'] - hard['oral_protrusion_psnr']:+.2f} dB** (free minus hard).

## Interpretation

The FLAME-distance `unsupported` region gives a strong static support bottleneck, and the recovery is continuous rather than binary. However, the directly audited oral-protrusion ROI improves by only **{free['oral_protrusion_psnr'] - hard['oral_protrusion_psnr']:+.2f} dB**. Therefore this run does **not** establish that hard FLAME means cannot reproduce tongue appearance. A hard-mean 3DGS can partly bypass its center constraint through learnable Gaussian extent/covariance. Scale-cap ablations are required before attributing the RGB curve to FLAME center support alone.

{f"On held-out cameras, the best oral condition is **{findings['regional_optima']['oral_heldout']}**, while the best broader unsupported condition is **{findings['regional_optima']['unsupported_heldout']}**. This is a single-case witness that the optimal prior strength can be region-dependent." if has_heldout else ""}

## Scope and caveats

- {'Twelve views are optimized and four angularly interleaved views are held out.' if has_heldout else 'All 16 views are optimized and evaluated; this is an oracle ceiling test, not novel-view generalization.'}
- Geometry is a 16-silhouette visual-hull proxy, not scan ground truth; mouth concavities are not identifiable.
- The FLAME-distance supported/unsupported partition is the primary region definition.
- `oral-protrusion` is an exploratory, visually audited polygon from the two mouth corners and a tongue-attracted chin landmark. It is not a substitute for manually verified semantic masks.
- This single subject/frame establishes a mechanism witness, not a population claim and not the animation-binding result.
"""
    (args.output / "findings.md").write_text(text)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
