#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from scipy.optimize import least_squares

from flame_oracle.flame import (
    apply_dynamic_flame,
    apply_tracking_world_transform,
    load_flame_dynamic_assets,
    rotation_matrix_from_rotvec,
)
from flame_oracle.io import (
    background_subtraction_alpha,
    load_multiview_frame,
    read_image,
    read_video_frame,
    save_montage,
)
from scripts.build_visual_hull import projection_overlay


DEFAULT_ASSET_ROOT = Path(
    "/data1/workspace/airulan/3dv/external/LAM/model_zoo/human_parametric_models/flame_vhap"
)


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (target - target_mean).T @ (source - source_mean) / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.asarray([1.0, 1.0, np.linalg.det(left @ right_t)])
    rotation = left @ np.diag(sign) @ right_t
    variance = np.mean(np.sum((source - source_mean) ** 2, axis=1))
    scale = float(np.dot(singular, sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def largest_face_landmarks(detector: object, image: np.ndarray) -> np.ndarray:
    detections = detector.get_landmarks(np.clip(image, 0, 255).astype(np.uint8))
    if not detections:
        raise RuntimeError("FaceAlignment failed to detect a face")
    candidates = [np.asarray(detection, dtype=np.float32)[:, :2] for detection in detections]
    areas = [float(np.prod(points.max(axis=0) - points.min(axis=0))) for points in candidates]
    return candidates[int(np.argmax(areas))]


def oral_aperture(landmarks: np.ndarray) -> float:
    mouth_width = float(np.linalg.norm(landmarks[54] - landmarks[48]))
    inner_height = float(np.linalg.norm(landmarks[66] - landmarks[62]))
    return inner_height / max(mouth_width, 1e-6)


def triangulate_landmark_robust(
    projection_matrices: list[np.ndarray],
    observations: np.ndarray,
    minimum_views: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(projection_matrices) < minimum_views:
        raise ValueError("too few views for triangulation")

    def dlt(matrices: list[np.ndarray], pixels: np.ndarray) -> np.ndarray:
        rows = []
        for matrix, (u, v) in zip(matrices, pixels):
            rows.extend([u * matrix[2] - matrix[0], v * matrix[2] - matrix[1]])
        _, _, vh = np.linalg.svd(np.stack(rows))
        return vh[-1, :3] / vh[-1, 3]

    def residual(point: np.ndarray, matrices: list[np.ndarray], pixels: np.ndarray) -> np.ndarray:
        homogeneous = np.r_[point, 1.0]
        values = []
        for matrix, target in zip(matrices, pixels):
            projected = matrix @ homogeneous
            values.extend(projected[:2] / projected[2] - target)
        return np.asarray(values)

    active = np.arange(len(projection_matrices), dtype=np.int64)
    point = dlt(projection_matrices, observations)
    for _ in range(3):
        matrices = [projection_matrices[index] for index in active]
        pixels = observations[active]
        point = least_squares(
            lambda value: residual(value, matrices, pixels),
            point,
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=300,
        ).x
        all_error = np.linalg.norm(
            residual(point, projection_matrices, observations).reshape(-1, 2), axis=1
        )
        active_error = all_error[active]
        median = float(np.median(active_error))
        mad = float(np.median(np.abs(active_error - median)))
        threshold = max(4.0, median + 3.0 * max(mad, 0.5))
        kept = active[active_error <= threshold]
        if len(kept) < minimum_views or np.array_equal(kept, active):
            break
        active = kept
    all_error = np.linalg.norm(
        residual(point, projection_matrices, observations).reshape(-1, 2), axis=1
    )
    return point.astype(np.float32), all_error.astype(np.float32), active


def vertices_to_landmarks(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    landmark_face_indices: torch.Tensor,
    landmark_barycentric: torch.Tensor,
) -> torch.Tensor:
    triangles = vertices[:, faces[landmark_face_indices]]
    return torch.sum(triangles * landmark_barycentric[None, :, :, None], dim=2)


def fit_dynamic_flame_landmarks(
    target_landmarks: np.ndarray,
    asset_root: Path,
    shape_dimensions: int,
    expression_dimensions: int,
    steps: int,
    coefficient_limit: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    assets_np = load_flame_dynamic_assets(
        asset_root / "flame2023.pkl",
        shape_dimensions=shape_dimensions,
        expression_dimensions=expression_dimensions,
    )
    embedding = np.load(
        asset_root / "landmark_embedding_with_eyes.npy", allow_pickle=True, encoding="latin1"
    ).item()
    landmark_face_indices_np = np.asarray(embedding["full_lmk_faces_idx"])[0, :68]
    landmark_barycentric_np = np.asarray(embedding["full_lmk_bary_coords"])[0, :68]
    device = torch.device("cuda")
    assets = {
        key: torch.from_numpy(value).to(device)
        for key, value in assets_np.items()
        if isinstance(value, np.ndarray)
    }
    faces = assets["faces"].long()
    landmark_face_indices = torch.from_numpy(landmark_face_indices_np).long().to(device)
    landmark_barycentric = torch.from_numpy(landmark_barycentric_np.astype(np.float32)).to(device)
    zeros_shape = torch.zeros((1, shape_dimensions), device=device)
    zeros_expression = torch.zeros((1, expression_dimensions), device=device)
    zeros_neck = torch.zeros((1, 3), device=device)
    zeros_jaw = torch.zeros((1, 3), device=device)
    zeros_eyes = torch.zeros((1, 6), device=device)
    with torch.no_grad():
        neutral_vertices = apply_dynamic_flame(
            assets, zeros_shape, zeros_expression, zeros_neck, zeros_jaw, zeros_eyes
        )
        neutral_landmarks = vertices_to_landmarks(
            neutral_vertices, faces, landmark_face_indices, landmark_barycentric
        )[0].cpu().numpy()

    stable_indices = np.r_[17:48, 48:68]
    scale_initial, rotation_initial, translation_initial = umeyama(
        neutral_landmarks[stable_indices], target_landmarks[stable_indices]
    )
    rotation_vector_initial, _ = cv2.Rodrigues(rotation_initial)
    shape = torch.nn.Parameter(zeros_shape.clone())
    expression = torch.nn.Parameter(zeros_expression.clone())
    neck = torch.nn.Parameter(zeros_neck.clone())
    jaw = torch.nn.Parameter(zeros_jaw.clone())
    eyes = torch.nn.Parameter(zeros_eyes.clone())
    global_rotvec = torch.nn.Parameter(
        torch.from_numpy(rotation_vector_initial[:, 0].astype(np.float32)).to(device)
    )
    log_scale = torch.nn.Parameter(
        torch.tensor(math.log(scale_initial), dtype=torch.float32, device=device)
    )
    translation = torch.nn.Parameter(
        torch.from_numpy(translation_initial.astype(np.float32)).to(device)
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [shape, expression], "lr": 1e-2},
            {"params": [neck, jaw, eyes], "lr": 3e-3},
            {"params": [global_rotvec, log_scale, translation], "lr": 2e-4},
        ]
    )
    target = torch.from_numpy(target_landmarks.astype(np.float32)).to(device)[None]
    weights = torch.ones((1, 68, 1), device=device)
    weights[:, :17] = 0.5
    weights[:, 27:48] = 2.0
    weights[:, 48:68] = 2.0
    history: list[dict[str, float | int]] = []

    def forward() -> tuple[torch.Tensor, torch.Tensor]:
        model_vertices = apply_dynamic_flame(assets, shape, expression, neck, jaw, eyes)
        rotation = rotation_matrix_from_rotvec(global_rotvec)[None]
        world_vertices = apply_tracking_world_transform(
            model_vertices, rotation, translation[None], torch.exp(log_scale)[None]
        )
        landmarks = vertices_to_landmarks(
            world_vertices, faces, landmark_face_indices, landmark_barycentric
        )
        return world_vertices, landmarks

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, predicted = forward()
        residual = torch.nn.functional.smooth_l1_loss(
            predicted, target, beta=0.002, reduction="none"
        )
        landmark_loss = (residual * weights).sum() / (weights.sum() * 3.0)
        prior = (
            2e-4 * shape.square().mean()
            + 5e-4 * expression.square().mean()
            + 2e-3 * neck.square().mean()
            + 1e-3 * jaw.square().mean()
            + 2e-3 * eyes.square().mean()
        )
        loss = landmark_loss + prior
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            shape.clamp_(-coefficient_limit, coefficient_limit)
            expression.clamp_(-coefficient_limit, coefficient_limit)
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == steps:
            with torch.no_grad():
                _, current = forward()
                errors = torch.linalg.vector_norm(current - target, dim=2)[0] * 1000.0
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "landmark_mean_mm": float(errors.mean()),
                    "landmark_median_mm": float(errors.median()),
                    "landmark_max_mm": float(errors.max()),
                }
            )

    with torch.no_grad():
        world_vertices, predicted = forward()
    result = {
        "vertices": world_vertices[0].cpu().numpy().astype(np.float32),
        "faces": assets_np["faces"].astype(np.int64),
        "predicted_landmarks": predicted[0].cpu().numpy().astype(np.float32),
        "shape": shape[0].detach().cpu().numpy().astype(np.float32),
        "expression": expression[0].detach().cpu().numpy().astype(np.float32),
        "neck": neck[0].detach().cpu().numpy().astype(np.float32),
        "jaw": jaw[0].detach().cpu().numpy().astype(np.float32),
        "eyes": eyes[0].detach().cpu().numpy().astype(np.float32),
        "global_rotvec": global_rotvec.detach().cpu().numpy().astype(np.float32),
        "scale": np.asarray([float(torch.exp(log_scale).detach().cpu())], dtype=np.float32),
        "translation": translation.detach().cpu().numpy().astype(np.float32),
    }
    report = {
        "shape_dimensions": shape_dimensions,
        "expression_dimensions": expression_dimensions,
        "steps": steps,
        "coefficient_limit": coefficient_limit,
        "shape_l2": float(np.linalg.norm(result["shape"])),
        "expression_l2": float(np.linalg.norm(result["expression"])),
        "jaw_l2": float(np.linalg.norm(result["jaw"])),
        "history": history,
    }
    return result, report


def project_vertex_colors(
    vertices: np.ndarray,
    images: np.ndarray,
    alpha: np.ndarray,
    viewmats: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    samples = np.full((len(images), len(vertices), 3), np.nan, dtype=np.float32)
    homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    for index, (image, a, viewmat, K) in enumerate(zip(images, alpha, viewmats, intrinsics)):
        camera = (viewmat @ homogeneous.T).T[:, :3]
        projected = (K @ camera.T).T
        pixels = projected[:, :2] / projected[:, 2:3].clip(min=1e-6)
        x = np.rint(pixels[:, 0]).astype(np.int64)
        y = np.rint(pixels[:, 1]).astype(np.int64)
        valid = (
            (camera[:, 2] > 0)
            & (x >= 0)
            & (x < image.shape[1])
            & (y >= 0)
            & (y < image.shape[0])
        )
        valid_indices = np.flatnonzero(valid)
        foreground = a[y[valid], x[valid], 0] > 0.5
        valid_indices = valid_indices[foreground]
        samples[index, valid_indices] = image[y[valid_indices], x[valid_indices]]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        colors = np.nanmedian(samples, axis=0)
    colors[~np.isfinite(colors)] = 0.5
    return colors.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("/data1/workspace/airulan/3dv/datasets/nersemble_v2"),
    )
    parser.add_argument("--sequence", default="FREE")
    parser.add_argument("--source-camera", default="222200037")
    parser.add_argument("--candidate-count", type=int, default=7)
    parser.add_argument("--fit-height", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=128)
    parser.add_argument("--background-threshold", type=int, default=30)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--shape-dimensions", type=int, default=100)
    parser.add_argument("--expression-dimensions", type=int, default=50)
    parser.add_argument("--fit-steps", type=int, default=1000)
    parser.add_argument("--coefficient-limit", type=float, default=3.0)
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Write a six-view identifiable visual audit; disabled for the bulk private cache.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    subject_root = args.v2_root / args.subject
    source_path = (
        subject_root / "sequences" / args.sequence / "images" / f"cam_{args.source_camera}.mp4"
    )
    background_path = (
        subject_root / "sequences" / "BACKGROUND" / f"image_{args.source_camera}.jpg"
    )
    capture = cv2.VideoCapture(str(source_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    if frame_count <= 0:
        raise RuntimeError(f"invalid source video: {source_path}")

    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        face_detector="sfd",
        flip_input=True,
        device="cuda",
    )
    candidate_indices = np.unique(
        np.rint(np.linspace(0.1, 0.9, args.candidate_count) * (frame_count - 1)).astype(np.int64)
    )
    source_background = read_image(background_path, args.fit_height)
    candidate_report = []
    for frame_index in candidate_indices:
        image = read_video_frame(source_path, int(frame_index), args.fit_height)
        try:
            landmarks = largest_face_landmarks(detector, image)
            aperture = oral_aperture(landmarks)
            status = "ok"
        except RuntimeError:
            aperture = -1.0
            status = "detector_failed"
        candidate_report.append(
            {"frame": int(frame_index), "oral_aperture_ratio": float(aperture), "status": status}
        )
    valid_candidates = [row for row in candidate_report if row["status"] == "ok"]
    if not valid_candidates:
        raise RuntimeError("FaceAlignment failed on every candidate source frame")
    chosen_frame = int(max(valid_candidates, key=lambda row: row["oral_aperture_ratio"])["frame"])

    case = load_multiview_frame(
        subject_root,
        args.sequence,
        chosen_frame,
        args.fit_height,
        background_threshold=args.background_threshold,
    )
    observations: list[np.ndarray | None] = []
    failed_cameras = []
    for camera_id, image in zip(case["camera_ids"], case["images"]):
        try:
            observations.append(largest_face_landmarks(detector, image * 255.0))
        except RuntimeError:
            observations.append(None)
            failed_cameras.append(camera_id)
    valid_indices = [index for index, value in enumerate(observations) if value is not None]
    if len(valid_indices) < 4:
        raise RuntimeError(f"only {len(valid_indices)} multiview landmark detections")
    projection_matrices = [
        K @ viewmat[:3, :4] for K, viewmat in zip(case["Ks"], case["viewmats"])
    ]
    target_landmarks = []
    triangulation_rows = []
    for landmark_index in range(68):
        matrices = [projection_matrices[index] for index in valid_indices]
        pixels = np.stack([observations[index][landmark_index] for index in valid_indices])
        point, errors, active = triangulate_landmark_robust(matrices, pixels)
        target_landmarks.append(point)
        triangulation_rows.append(
            {
                "landmark": landmark_index,
                "input_views": len(valid_indices),
                "kept_views": int(len(active)),
                "median_reprojection_px": float(np.median(errors[active])),
                "max_kept_reprojection_px": float(np.max(errors[active])),
            }
        )
    target_landmarks_np = np.stack(target_landmarks).astype(np.float32)
    fit, fit_report = fit_dynamic_flame_landmarks(
        target_landmarks_np,
        args.asset_root,
        args.shape_dimensions,
        args.expression_dimensions,
        args.fit_steps,
        args.coefficient_limit,
    )

    scale = args.render_height / args.fit_height
    output_width = round(int(case["width"]) * scale)
    images_small = np.stack(
        [cv2.resize(image, (output_width, args.render_height), interpolation=cv2.INTER_AREA) for image in case["images"]]
    ).astype(np.float32)
    alpha_small = np.stack(
        [cv2.resize(a, (output_width, args.render_height), interpolation=cv2.INTER_AREA) for a in case["alpha"]]
    )[..., None].astype(np.float32)
    if alpha_small.ndim == 5:
        alpha_small = alpha_small[..., 0]
    intrinsics_small = case["Ks"].copy()
    intrinsics_small[:, 0, :] *= scale
    intrinsics_small[:, 1, :] *= scale
    initial_colors = project_vertex_colors(
        fit["vertices"], images_small, alpha_small, case["viewmats"], intrinsics_small
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        subject=np.asarray(args.subject),
        sequence=np.asarray(args.sequence),
        frame=np.asarray([chosen_frame], dtype=np.int64),
        camera_ids=np.asarray(case["camera_ids"]),
        images=np.clip(images_small * 255.0, 0, 255).astype(np.uint8),
        alpha=np.clip(alpha_small * 255.0, 0, 255).astype(np.uint8),
        Ks=intrinsics_small.astype(np.float32),
        viewmats=case["viewmats"].astype(np.float32),
        vertices=fit["vertices"],
        faces=fit["faces"],
        initial_colors=initial_colors,
        target_landmarks=target_landmarks_np,
        predicted_landmarks=fit["predicted_landmarks"],
        shape=fit["shape"],
        expression=fit["expression"],
        neck=fit["neck"],
        jaw=fit["jaw"],
        eyes=fit["eyes"],
        global_rotvec=fit["global_rotvec"],
        scale=fit["scale"],
        translation=fit["translation"],
    )
    if args.save_overlay:
        overlays = [
            projection_overlay(image, fit["vertices"], viewmat, K)
            for image, viewmat, K in zip(
                case["images"][:6], case["viewmats"][:6], case["Ks"][:6]
            )
        ]
        save_montage(args.output.with_name(args.output.stem + "_overlay.png"), overlays, columns=3)
    landmark_error_mm = np.linalg.norm(
        fit["predicted_landmarks"] - target_landmarks_np, axis=1
    ) * 1000.0
    report = {
        "kind": "nersemble_v2_multiview_landmark_flame_scale_cache",
        "subject": args.subject,
        "sequence": args.sequence,
        "source_camera": args.source_camera,
        "source_frame_count": frame_count,
        "candidate_frame_selection": candidate_report,
        "chosen_frame": chosen_frame,
        "chosen_oral_aperture_ratio": max(
            valid_candidates, key=lambda row: row["oral_aperture_ratio"]
        )["oral_aperture_ratio"],
        "camera_ids": case["camera_ids"],
        "landmark_detector_failed_cameras": failed_cameras,
        "landmark_detector_usable_views": len(valid_indices),
        "triangulation": {
            "median_landmark_median_reprojection_px": float(
                np.median([row["median_reprojection_px"] for row in triangulation_rows])
            ),
            "max_landmark_median_reprojection_px": float(
                np.max([row["median_reprojection_px"] for row in triangulation_rows])
            ),
            "per_landmark": triangulation_rows,
        },
        "fit": {
            **fit_report,
            "landmark_mean_mm": float(landmark_error_mm.mean()),
            "landmark_median_mm": float(np.median(landmark_error_mm)),
            "landmark_max_mm": float(landmark_error_mm.max()),
        },
        "cache": {
            "output": str(args.output),
            "resolution": [args.render_height, output_width],
            "camera_count": len(case["camera_ids"]),
            "alpha_source": case["alpha_source"],
        },
        "scope_warning": (
            "One static multiview frame selected from FREE by source-view oral aperture. The fitted FLAME "
            "scaffold uses multiview 68-landmark triangulation and is therefore an oracle scaffold for "
            "the static N_id proxy, not deployable single-view tracking and not dynamic supervision."
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "subject": args.subject,
                "chosen_frame": chosen_frame,
                "usable_views": len(valid_indices),
                "triangulation_median_px": report["triangulation"][
                    "median_landmark_median_reprojection_px"
                ],
                "fit_mean_mm": report["fit"]["landmark_mean_mm"],
                "fit_max_mm": report["fit"]["landmark_max_mm"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
