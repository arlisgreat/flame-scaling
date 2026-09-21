#!/usr/bin/env python3
"""Post-audit decomposition of unsupported head content versus torso/shoulders.

The original pre-registered ``flame_unsupported`` mask is retained untouched.
This evaluator loads frozen checkpoints and adds a disclosed head-envelope
analysis after qualitative audit showed that the broad foreground mask contains
substantial clothing and shoulders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import lpips
import numpy as np
import torch

from flame_oracle.io import save_montage
from flame_oracle.metrics import alpha_metrics, masked_image_metrics
from scripts.run_identity_scaling import (
    DEFAULT_CACHE,
    DEFAULT_MANIFEST,
    DEFAULT_MASKS,
    SharedGaussianReconstructor,
    augment_subject_density,
    load_subjects,
    mask_region_indices,
    mesh_silhouette,
    predict_attributes,
    project_points,
    render_attributes,
    surface_sampling_plan,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head_envelope(
    subject: dict[str, Any],
    camera_index: int,
    head_indices: np.ndarray,
    dilation_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    viewmat = subject["viewmats"][camera_index].detach().cpu().numpy()
    intrinsic = subject["Ks"][camera_index].detach().cpu().numpy()
    vertices = subject["base"].detach().cpu().numpy()
    height, width = subject["height"], subject["width"]
    pixels, depth = project_points(vertices[head_indices], viewmat, intrinsic)
    visible = (depth > 0) & np.isfinite(pixels).all(axis=1)
    envelope = np.zeros((height, width), dtype=np.uint8)
    if visible.sum() >= 3:
        hull = cv2.convexHull(np.rint(pixels[visible]).astype(np.int32))
        cv2.fillConvexPoly(envelope, hull, 1)
    if dilation_px > 0:
        size = 2 * dilation_px + 1
        envelope = cv2.dilate(
            envelope, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        )
    support = mesh_silhouette(
        vertices,
        subject["faces"],
        viewmat,
        intrinsic,
        height,
        width,
    )
    support = cv2.dilate(
        support.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool)
    return envelope.astype(bool), support


def region_masks(
    subject: dict[str, Any],
    camera_index: int,
    target_alpha: np.ndarray,
    head_indices: np.ndarray,
    dilation_px: int,
) -> dict[str, np.ndarray]:
    head_roi, support = head_envelope(
        subject, camera_index, head_indices, dilation_px
    )
    foreground = target_alpha[..., 0] > 0.05
    return {
        "head_foreground": foreground & head_roi,
        "head_supported": foreground & head_roi & support,
        "head_unsupported": foreground & head_roi & ~support,
        "torso_outside_head_roi": foreground & ~head_roi,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--head-dilation-px", type=int, default=6)
    parser.add_argument("--audit-subjects", nargs="*", default=[])
    args = parser.parse_args()
    if args.head_dilation_px < 0:
        parser.error("--head-dilation-px must be non-negative")
    metrics_path = args.run / "metrics.json"
    checkpoint_path = args.run / "checkpoint.pt"
    original = json.loads(metrics_path.read_text())
    manifest = json.loads(args.manifest.read_text())
    device = torch.device("cuda")
    common_camera_ids = original["camera_ids"]
    test_subjects = load_subjects(
        original["test_subjects"], args.cache_root, common_camera_ids, device
    )
    geometry = original["geometry_intervention"]
    density_multiplier = int(geometry.get("density_multiplier", 1))
    if density_multiplier > 1:
        original_vertex_count = int(
            geometry.get("original_vertex_count", len(test_subjects[0]["base"]))
        )
        face_indices, barycentric = surface_sampling_plan(
            test_subjects[0]["faces"],
            original_vertex_count,
            density_multiplier,
            int(geometry["surface_sampling_seed"]),
        )
        for subject in test_subjects:
            augment_subject_density(subject, face_indices, barycentric)
    gaussian_count = original["geometry_intervention"]["gaussian_count"]
    if any(len(subject["base"]) != gaussian_count for subject in test_subjects):
        raise RuntimeError("reconstructed density does not match checkpoint Gaussian count")
    architecture = original["architecture"]
    model = SharedGaussianReconstructor(
        gaussian_count,
        architecture["latent_dim"],
        architecture["slot_dim"],
        architecture["hidden_dim"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    perceptual = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    camera_order = [int(value) for value in checkpoint["camera_order"]]
    groups = mask_region_indices(args.mask_path)
    head_indices = np.unique(
        np.concatenate(
            [groups[key] for key in ["face", "oral_slots", "eye_slots", "scalp", "ears"]]
        )
    )
    rows: list[dict[str, Any]] = []
    mask_counts: list[dict[str, Any]] = []
    with torch.no_grad():
        for subject in test_subjects:
            for n_obs in original["observation_counts"]:
                observations = camera_order[:n_obs]
                targets = camera_order[n_obs:]
                attributes = predict_attributes(
                    model,
                    subject,
                    observations,
                    original["method"],
                    original["geometry_intervention"]["adaptive_radius_mm"],
                    original["geometry_intervention"]["released_radius_mm"],
                )
                target_tensor = torch.as_tensor(targets, device=device)
                rendered, predicted_alpha = render_attributes(
                    attributes,
                    subject["viewmats"][target_tensor],
                    subject["Ks"][target_tensor],
                    subject["width"],
                    subject["height"],
                    original["geometry_intervention"]["fixed_isotropic_scale_mm"],
                )
                images_t = subject["images_u8"][target_tensor].float() / 255.0
                alpha_t = subject["alpha_u8"][target_tensor].float() / 255.0
                target_t = images_t * alpha_t + (1.0 - alpha_t)
                perceptual_map = perceptual(
                    rendered.permute(0, 3, 1, 2) * 2.0 - 1.0,
                    target_t.permute(0, 3, 1, 2) * 2.0 - 1.0,
                ).detach().cpu().numpy()[:, 0]
                rendered_np = rendered.cpu().numpy()
                predicted_alpha_np = predicted_alpha.cpu().numpy()
                target_np = target_t.cpu().numpy()
                alpha_np = alpha_t.cpu().numpy()
                audit_panels: list[np.ndarray] = []
                for local_index, camera_index in enumerate(targets):
                    masks = region_masks(
                        subject,
                        camera_index,
                        alpha_np[local_index],
                        head_indices,
                        args.head_dilation_px,
                    )
                    counts = {
                        "subject": subject["subject"],
                        "n_obs": n_obs,
                        "camera_id": common_camera_ids[camera_index],
                        **{f"{name}_pixels": int(mask.sum()) for name, mask in masks.items()},
                    }
                    mask_counts.append(counts)
                    for region, mask in masks.items():
                        rows.append(
                            {
                                "subject": subject["subject"],
                                "n_obs": n_obs,
                                "camera_id": common_camera_ids[camera_index],
                                "region": region,
                                "pixel_count": int(mask.sum()),
                                "lpips_alex_spatial": float(
                                    (perceptual_map[local_index] * mask).sum()
                                    / max(float(mask.sum()), 1.0)
                                ),
                                **masked_image_metrics(
                                    rendered_np[local_index], target_np[local_index], mask
                                ),
                                **alpha_metrics(
                                    predicted_alpha_np[local_index] * mask[..., None],
                                    alpha_np[local_index] * mask[..., None],
                                ),
                            }
                        )
                    if (
                        subject["subject"] in set(args.audit_subjects)
                        and n_obs == 4
                        and local_index < 6
                    ):
                        overlay = target_np[local_index].copy()
                        overlay[masks["head_supported"]] = (
                            0.55 * overlay[masks["head_supported"]]
                            + 0.45 * np.asarray([0.1, 0.9, 0.2])
                        )
                        overlay[masks["head_unsupported"]] = (
                            0.55 * overlay[masks["head_unsupported"]]
                            + 0.45 * np.asarray([0.95, 0.15, 0.1])
                        )
                        overlay[masks["torso_outside_head_roi"]] = (
                            0.55 * overlay[masks["torso_outside_head_roi"]]
                            + 0.45 * np.asarray([0.1, 0.35, 0.95])
                        )
                        audit_panels.extend(
                            [target_np[local_index], overlay, rendered_np[local_index]]
                        )
                if audit_panels:
                    save_montage(
                        args.run
                        / "head_roi_audits"
                        / f"{subject['subject']}_nobs4_target_mask_prediction.png",
                        audit_panels,
                        columns=3,
                    )
    write_csv(args.run / "head_roi_rows.csv", rows)
    report = {
        "kind": "post_audit_head_roi_identity_scaling_evaluation",
        "source_metrics": str(metrics_path),
        "source_metrics_sha256": sha256(metrics_path),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "method": original["method"],
        "n_id": original["n_id"],
        "seed": original["seed"],
        "test_subjects": original["test_subjects"],
        "test_rows": rows,
        "mask_counts": mask_counts,
        "head_roi_definition": {
            "vertices": "convex hull of FLAME face, lips, eye regions, scalp, and ears; neck excluded",
            "dilation_px_at_128px_height": args.head_dilation_px,
            "support": "projected full FLAME mesh silhouette dilated by one pixel",
            "head_unsupported": "target foreground AND dilated head hull AND NOT support",
            "torso_outside_head_roi": "target foreground outside dilated head hull",
        },
        "status": "post-audit decomposition; original pre-registered metrics remain primary and unchanged",
        "reason": (
            "Qualitative audit after the frozen broad analysis showed that flame_unsupported includes "
            "substantial clothing and shoulders. This decomposition tests whether the same effect remains "
            "inside a head-local envelope."
        ),
    }
    (args.run / "head_roi_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run": str(args.run),
                "rows": len(rows),
                "head_dilation_px": args.head_dilation_px,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
