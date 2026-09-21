#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from flame_oracle.flame import apply_linear_flame, load_flame_linear_assets
from flame_oracle.io import load_multiview_frame, read_point_cloud, save_montage
from scripts.build_visual_hull import projection_overlay


DEFAULT_ASSET_ROOT = Path(
    "/data1/workspace/airulan/3dv/external/LAM/model_zoo/human_parametric_models/flame_vhap"
)


def detect_stable_landmarks(image: np.ndarray, detector: object) -> dict[str, np.ndarray]:
    result = detector.process(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    if not result.multi_face_landmarks:
        raise RuntimeError("MediaPipe failed to detect a face")
    points = result.multi_face_landmarks[0].landmark
    # Deliberately exclude chin and lower-lip landmarks: on protruding-tongue
    # frames MediaPipe consistently places them on the tongue tip.
    return {
        "left_eye": np.mean([[points[362].x, points[362].y], [points[263].x, points[263].y]], axis=0),
        "right_eye": np.mean([[points[33].x, points[33].y], [points[133].x, points[133].y]], axis=0),
        "nose_tip": np.asarray([points[1].x, points[1].y]),
        "left_mouth": np.asarray([points[291].x, points[291].y]),
        "right_mouth": np.asarray([points[61].x, points[61].y]),
        "upper_lip": np.asarray([points[13].x, points[13].y]),
    }


def triangulate_robust(
    projection_matrices: list[np.ndarray],
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for matrix, (u, v) in zip(projection_matrices, observations):
        rows.extend([u * matrix[2] - matrix[0], v * matrix[2] - matrix[1]])
    _, _, vh = np.linalg.svd(np.stack(rows))
    initial = vh[-1, :3] / vh[-1, 3]

    def residual(point: np.ndarray) -> np.ndarray:
        values = []
        homogeneous = np.r_[point, 1.0]
        for matrix, target in zip(projection_matrices, observations):
            projected = matrix @ homogeneous
            values.extend(projected[:2] / projected[2] - target)
        return np.asarray(values)

    point = least_squares(
        residual,
        initial,
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=500,
    ).x
    error = np.linalg.norm(residual(point).reshape(-1, 2), axis=1)
    return point.astype(np.float32), error.astype(np.float32)


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


def semantic_definition(template: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    lips_ids = np.asarray(masks["lips"], dtype=np.int64)
    lips = template[lips_ids]
    central = np.abs(lips[:, 0]) < 0.003
    upper_lip_id = lips_ids[int(np.argmax(np.where(central, lips[:, 1], -np.inf)))]
    nose_ids = np.asarray(masks["nose"], dtype=np.int64)
    nose_tip_id = nose_ids[int(np.argmax(template[nose_ids, 2]))]
    return {
        "left_eye": np.asarray(masks["left_eyeball"], dtype=np.int64),
        "right_eye": np.asarray(masks["right_eyeball"], dtype=np.int64),
        "nose_tip": np.asarray([nose_tip_id], dtype=np.int64),
        "left_mouth": np.asarray([lips_ids[int(np.argmax(lips[:, 0]))]], dtype=np.int64),
        "right_mouth": np.asarray([lips_ids[int(np.argmin(lips[:, 0]))]], dtype=np.int64),
        "upper_lip": np.asarray([upper_lip_id], dtype=np.int64),
    }


def semantic_points(vertices: torch.Tensor, definition: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([vertices[definition[name]].mean(dim=0) for name in definition])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--shape-dims", type=int, default=100)
    parser.add_argument("--expression-dims", type=int, default=50)
    parser.add_argument("--outer-iterations", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=200)
    parser.add_argument("--max-correspondence-mm", type=float, default=35.0)
    parser.add_argument("--exclude-mouth-radius-mm", type=float, default=45.0)
    parser.add_argument("--surface-weight", type=float, default=1.0)
    parser.add_argument("--landmark-weight", type=float, default=3.0)
    parser.add_argument("--shape-prior", type=float, default=1e-3)
    parser.add_argument("--expression-prior", type=float, default=1e-3)
    parser.add_argument("--coefficient-limit", type=float, default=3.0)
    parser.add_argument("--target-y-min", type=float, default=-0.16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case = load_multiview_frame(
        args.subject_root,
        args.sequence,
        args.frame,
        args.height,
        background_threshold=30,
    )
    observations: list[dict[str, np.ndarray] | None] = []
    detector_failed_cameras = []
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as detector:
        for camera_id, image in zip(case["camera_ids"], case["images"]):
            try:
                observations.append(detect_stable_landmarks(image, detector))
            except RuntimeError:
                observations.append(None)
                detector_failed_cameras.append(camera_id)

    valid_observations = [item for item in observations if item is not None]
    if len(valid_observations) < 3:
        raise RuntimeError(
            f"MediaPipe produced only {len(valid_observations)} usable views; need at least 3"
        )

    projection_matrices = [
        K @ viewmat[:3, :4] for K, viewmat in zip(case["Ks"], case["viewmats"])
    ]
    names = list(valid_observations[0])
    triangulated = {}
    triangulation_report = {}
    for name in names:
        valid = [
            (camera_id, matrix, item)
            for camera_id, matrix, item in zip(
                case["camera_ids"], projection_matrices, observations
            )
            if item is not None
        ]
        pixels = np.stack(
            [
                item[name] * np.asarray([case["width"], case["height"]], dtype=np.float32)
                for _, _, item in valid
            ]
        )
        point, errors = triangulate_robust([matrix for _, matrix, _ in valid], pixels)
        triangulated[name] = point
        triangulation_report[name] = {
            "world_xyz": point.tolist(),
            "median_reprojection_px": float(np.median(errors)),
            "mean_reprojection_px": float(errors.mean()),
            "max_reprojection_px": float(errors.max()),
            "per_camera_reprojection_px": {
                camera_id: float(error)
                for (camera_id, _, _), error in zip(valid, errors)
            },
        }

    assets = load_flame_linear_assets(
        args.asset_root / "flame2023.pkl",
        args.asset_root / "FLAME_masks.pkl",
        args.shape_dims,
        args.expression_dims,
    )
    template = assets["template"]
    directions = assets["directions"]
    faces = assets["faces"]
    masks = assets["masks"]
    definition_np = semantic_definition(template, masks)
    source_semantic = np.stack([template[definition_np[name]].mean(axis=0) for name in names])
    target_semantic = np.stack([triangulated[name] for name in names])
    scale_initial, rotation_initial, translation_initial = umeyama(source_semantic, target_semantic)

    pcd = read_point_cloud(args.pcd)
    target = pcd["xyz"][pcd["xyz"][:, 1] >= args.target_y_min]
    target_tree = cKDTree(target)
    fit_ids = np.unique(
        np.concatenate([masks["face"], masks["left_ear"], masks["right_ear"], masks["scalp"]])
    )
    distance_to_lips = cKDTree(template[masks["lips"]]).query(template, workers=-1)[0]
    mouth_excluded = distance_to_lips <= args.exclude_mouth_radius_mm / 1000.0
    fit_ids = fit_ids[~mouth_excluded[fit_ids]]

    device = torch.device("cuda")
    template_t = torch.from_numpy(template).to(device)
    directions_t = torch.from_numpy(directions).to(device)
    coefficients = torch.nn.Parameter(torch.zeros(directions.shape[-1], device=device))
    rotvec = torch.nn.Parameter(
        torch.from_numpy(Rotation.from_matrix(rotation_initial).as_rotvec().astype(np.float32)).to(device)
    )
    log_scale = torch.nn.Parameter(
        torch.tensor(math.log(scale_initial), dtype=torch.float32, device=device)
    )
    translation = torch.nn.Parameter(torch.from_numpy(translation_initial).to(device))
    optimizer = torch.optim.Adam(
        [
            {"params": [coefficients], "lr": 1e-2},
            {"params": [rotvec], "lr": 1e-4},
            {"params": [log_scale], "lr": 1e-4},
            {"params": [translation], "lr": 1e-4},
        ]
    )
    fit_ids_t = torch.from_numpy(fit_ids).long().to(device)
    definition_t = {
        name: torch.from_numpy(ids).long().to(device) for name, ids in definition_np.items()
    }
    target_semantic_t = torch.from_numpy(target_semantic.astype(np.float32)).to(device)
    history = []
    max_distance = args.max_correspondence_mm / 1000.0
    for outer in range(args.outer_iterations):
        with torch.no_grad():
            current = apply_linear_flame(
                template_t, directions_t, coefficients, rotvec, log_scale, translation
            )
            current_fit = current[fit_ids_t].cpu().numpy()
        distances, indices = target_tree.query(current_fit, workers=-1)
        keep = distances < max_distance
        if int(keep.sum()) < 500:
            raise RuntimeError(f"too few stable correspondences at outer {outer}: {keep.sum()}")
        kept_ids = torch.from_numpy(fit_ids[keep]).long().to(device)
        matched = torch.from_numpy(target[indices[keep]].astype(np.float32)).to(device)

        for _ in range(args.inner_steps):
            optimizer.zero_grad(set_to_none=True)
            current = apply_linear_flame(
                template_t, directions_t, coefficients, rotvec, log_scale, translation
            )
            surface_loss = torch.nn.functional.smooth_l1_loss(
                current[kept_ids], matched, beta=0.003
            )
            predicted_semantic = semantic_points(current, definition_t)
            landmark_loss = torch.nn.functional.smooth_l1_loss(
                predicted_semantic, target_semantic_t, beta=0.003
            )
            shape = coefficients[: args.shape_dims]
            expression = coefficients[args.shape_dims :]
            prior = args.shape_prior * shape.square().mean()
            if expression.numel():
                prior = prior + args.expression_prior * expression.square().mean()
            loss = (
                args.surface_weight * surface_loss
                + args.landmark_weight * landmark_loss
                + prior
            )
            loss.backward()
            optimizer.step()
            if args.coefficient_limit > 0:
                with torch.no_grad():
                    coefficients.clamp_(-args.coefficient_limit, args.coefficient_limit)

        with torch.no_grad():
            current = apply_linear_flame(
                template_t, directions_t, coefficients, rotvec, log_scale, translation
            )
            semantic_error = (
                (semantic_points(current, definition_t) - target_semantic_t).norm(dim=1) * 1000.0
            )
            post_surface = target_tree.query(current[fit_ids_t].cpu().numpy(), workers=-1)[0]
        record = {
            "outer": outer,
            "kept_correspondences": int(keep.sum()),
            "stable_surface_mean_mm": float(post_surface.mean() * 1000.0),
            "stable_surface_median_mm": float(np.median(post_surface) * 1000.0),
            "semantic_mean_mm": float(semantic_error.mean().cpu()),
            "semantic_max_mm": float(semantic_error.max().cpu()),
        }
        history.append(record)
        print(json.dumps(record))

    with torch.no_grad():
        vertices = apply_linear_flame(
            template_t, directions_t, coefficients, rotvec, log_scale, translation
        ).cpu().numpy()
        predicted_semantic = semantic_points(current, definition_t).cpu().numpy()
    full_tree = cKDTree(pcd["xyz"])
    nearest_distance, nearest_index = full_tree.query(vertices, workers=-1)
    initial_colors = pcd["rgb"][nearest_index]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int64),
        coefficients=coefficients.detach().cpu().numpy(),
        rotvec=rotvec.detach().cpu().numpy(),
        scale=np.asarray([float(torch.exp(log_scale).detach().cpu())], dtype=np.float32),
        translation=translation.detach().cpu().numpy(),
        initial_colors=initial_colors,
        nearest_pcd_distance=nearest_distance.astype(np.float32),
        semantic_names=np.asarray(names),
        semantic_target=target_semantic.astype(np.float32),
        semantic_predicted=predicted_semantic.astype(np.float32),
    )
    overlays = [
        projection_overlay(image, vertices, viewmat, K)
        for image, viewmat, K in zip(case["images"], case["viewmats"], case["Ks"])
    ]
    save_montage(args.output.with_name(args.output.stem + "_overlay.png"), overlays, columns=4)

    semantic_errors = np.linalg.norm(predicted_semantic - target_semantic, axis=1) * 1000.0
    report = {
        "kind": "multiview_stable_landmark_anchored_flame_fit",
        "pcd": str(args.pcd),
        "subject_root": str(args.subject_root),
        "sequence": args.sequence,
        "frame": args.frame,
        "camera_ids": case["camera_ids"],
        "landmark_detector_failed_cameras": detector_failed_cameras,
        "landmark_detector_usable_views": len(valid_observations),
        "excluded_unstable_landmarks": ["chin", "lower_lip"],
        "triangulation": triangulation_report,
        "initial_similarity": {
            "scale": scale_initial,
            "rotation": rotation_initial.tolist(),
            "translation": translation_initial.tolist(),
        },
        "fit": {
            "stable_fit_vertices": int(len(fit_ids)),
            "excluded_mouth_vertices": int(mouth_excluded.sum()),
            "surface_weight": args.surface_weight,
            "landmark_weight": args.landmark_weight,
            "shape_prior": args.shape_prior,
            "expression_prior": args.expression_prior,
            "coefficient_limit": args.coefficient_limit,
            "coefficient_l2": float(coefficients.detach().norm().cpu()),
            "coefficient_max_abs": float(coefficients.detach().abs().max().cpu()),
        },
        "history": history,
        "final_semantic_error_mm": {
            name: float(error) for name, error in zip(names, semantic_errors)
        },
        "warning": (
            "Stable-landmark oracle scaffold fit; MediaPipe is used only for initialization/anchoring. "
            "This is not the tracker-quality experiment."
        ),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "report": str(report_path), **report["fit"], "final_semantic_error_mm": report["final_semantic_error_mm"]}, indent=2))


if __name__ == "__main__":
    main()
