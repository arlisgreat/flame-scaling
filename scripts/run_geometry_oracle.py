#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from scipy.spatial import cKDTree

from flame_oracle.io import load_multiview_frame, read_point_cloud, save_montage, save_rgb
from flame_oracle.metrics import alpha_metrics, masked_image_metrics
from flame_oracle.regions import build_distance_region_masks
from flame_oracle.render import quaternions_from_z, vertex_normals


def logit(value: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    value = np.clip(value, epsilon, 1.0 - epsilon)
    return np.log(value / (1.0 - value))


def means_for_constraint(base: torch.Tensor, raw_delta: torch.Tensor, radius_mm: float) -> torch.Tensor:
    if radius_mm == 0:
        # Keep a dormant parameter tensor so all conditions have identical
        # total parameter counts while hard geometry has zero active DoF.
        return base + 0.0 * raw_delta
    if radius_mm < 0:
        return base + raw_delta
    norm = raw_delta.norm(dim=-1, keepdim=True)
    bounded = raw_delta / torch.sqrt(1.0 + norm.square())
    return base + (radius_mm / 1000.0) * bounded


def _farthest_point_order(vertices: np.ndarray) -> np.ndarray:
    """Deterministic full FPS ordering used to make low-density levels nested."""
    vertices64 = np.asarray(vertices, dtype=np.float64)
    count = len(vertices64)
    if count == 0:
        raise ValueError("vertices must be non-empty")
    order = np.empty(count, dtype=np.int64)
    order[0] = int(np.argmax(np.sum((vertices64 - vertices64.mean(axis=0)) ** 2, axis=1)))
    minimum_distance2 = np.sum((vertices64 - vertices64[order[0]]) ** 2, axis=1)
    minimum_distance2[order[0]] = -1.0
    for index in range(1, count):
        chosen = int(np.argmax(minimum_distance2))
        order[index] = chosen
        distance2 = np.sum((vertices64 - vertices64[chosen]) ** 2, axis=1)
        minimum_distance2 = np.minimum(minimum_distance2, distance2)
        minimum_distance2[order[: index + 1]] = -1.0
    return order


def build_density_scaffold(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    multiplier: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Build a deterministic nested FLAME-surface scaffold at a requested density.

    Below 1x, points are a prefix of a full farthest-point ordering of FLAME
    vertices. Above 1x, all original vertices are retained and extra points are
    sampled area-proportionally on triangles; the fixed RNG stream makes every
    larger level contain the same prefix of surface samples.
    """
    if multiplier <= 0:
        raise ValueError("density multiplier must be positive")
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(colors, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if len(vertices) != len(colors) or len(vertices) != len(normals):
        raise ValueError("vertices, colors, and normals must have equal length")
    target_count = max(1, int(round(len(vertices) * multiplier)))
    if target_count == len(vertices):
        return (
            vertices.copy(),
            colors.copy(),
            normals.copy(),
            {
                "multiplier": float(multiplier),
                "target_count": target_count,
                "original_vertex_count": int(len(vertices)),
                "vertex_count": int(len(vertices)),
                "surface_sample_count": 0,
                "construction": "original_vertices",
                "seed": int(seed),
            },
        )
    if target_count < len(vertices):
        selected = _farthest_point_order(vertices)[:target_count]
        return (
            vertices[selected].copy(),
            colors[selected].copy(),
            normals[selected].copy(),
            {
                "multiplier": float(multiplier),
                "target_count": target_count,
                "original_vertex_count": int(len(vertices)),
                "vertex_count": target_count,
                "surface_sample_count": 0,
                "construction": "nested_vertex_fps_prefix",
                "seed": int(seed),
            },
        )

    extra_count = target_count - len(vertices)
    triangles = vertices[faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    ).astype(np.float64)
    if not np.isfinite(double_area).all() or double_area.sum() <= 0:
        raise ValueError("mesh has no valid surface area")
    probabilities = double_area / double_area.sum()
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(faces), size=extra_count, replace=True, p=probabilities)
    random_uv = rng.random((extra_count, 2))
    sqrt_u = np.sqrt(random_uv[:, 0])
    barycentric = np.stack(
        [1.0 - sqrt_u, sqrt_u * (1.0 - random_uv[:, 1]), sqrt_u * random_uv[:, 1]], axis=1
    ).astype(np.float32)
    sampled_faces = faces[face_indices]
    sampled_vertices = np.sum(vertices[sampled_faces] * barycentric[:, :, None], axis=1)
    sampled_colors = np.sum(colors[sampled_faces] * barycentric[:, :, None], axis=1)
    sampled_normals = np.sum(normals[sampled_faces] * barycentric[:, :, None], axis=1)
    sampled_normals /= np.linalg.norm(sampled_normals, axis=1, keepdims=True).clip(min=1e-8)
    return (
        np.concatenate([vertices, sampled_vertices], axis=0).astype(np.float32),
        np.concatenate([colors, sampled_colors], axis=0).astype(np.float32),
        np.concatenate([normals, sampled_normals], axis=0).astype(np.float32),
        {
            "multiplier": float(multiplier),
            "target_count": target_count,
            "original_vertex_count": int(len(vertices)),
            "vertex_count": int(len(vertices)),
            "surface_sample_count": int(extra_count),
            "construction": "original_vertices_plus_nested_area_samples",
            "seed": int(seed),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument(
        "--subject-root",
        type=Path,
        default=Path("/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs/388"),
    )
    parser.add_argument("--sequence", default="GLASSES")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--pcd",
        type=Path,
        help="Optional point-cloud override; accepts NeRSemble PCD or oracle-generated NPZ.",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--background-threshold", type=int, default=30)
    parser.add_argument(
        "--heldout-camera-ids",
        nargs="*",
        default=[],
        help="Cameras excluded from optimization but rendered and reported separately.",
    )
    parser.add_argument("--radius-mm", type=float, required=True, help="0=hard, negative=free")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--position-lr",
        type=float,
        default=5e-4,
        help="Initial learning rate measured in world-coordinate meters for every non-hard condition.",
    )
    parser.add_argument(
        "--max-scale-mm",
        type=float,
        default=30.0,
        help="Hard upper bound for every Gaussian principal-axis standard deviation.",
    )
    parser.add_argument(
        "--scale-penalty-threshold-mm",
        type=float,
        default=12.0,
        help="Soft quadratic penalty starts above this principal-axis scale.",
    )
    parser.add_argument(
        "--scale-penalty-weight",
        type=float,
        default=100.0,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--density-multiplier",
        type=float,
        default=1.0,
        help="Gaussian scaffold cardinality relative to the 5,023 FLAME vertices.",
    )
    parser.add_argument(
        "--density-seed",
        type=int,
        default=20260715,
        help="Fixed surface-sampling seed shared by every density level.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_scale_mm <= 0:
        parser.error("--max-scale-mm must be positive")
    if args.scale_penalty_threshold_mm < 0:
        parser.error("--scale-penalty-threshold-mm must be non-negative")
    if args.density_multiplier <= 0:
        parser.error("--density-multiplier must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    fit = np.load(args.fit)
    flame_vertices_np = fit["vertices"].astype(np.float32)
    faces = fit["faces"].astype(np.int64)
    flame_colors_np = fit["initial_colors"].astype(np.float32) / 255.0
    flame_normals_np = vertex_normals(flame_vertices_np, faces)
    base_np, colors_np, normals_np, density_report = build_density_scaffold(
        flame_vertices_np,
        faces,
        flame_colors_np,
        flame_normals_np,
        args.density_multiplier,
        args.density_seed,
    )

    case = load_multiview_frame(
        args.subject_root,
        args.sequence,
        args.frame,
        args.height,
        background_threshold=args.background_threshold,
    )
    unknown_heldout = sorted(set(args.heldout_camera_ids) - set(case["camera_ids"]))
    if unknown_heldout:
        parser.error(f"unknown held-out camera ids: {unknown_heldout}")
    heldout_indices = [
        index for index, camera_id in enumerate(case["camera_ids"])
        if camera_id in set(args.heldout_camera_ids)
    ]
    train_indices = [
        index for index, camera_id in enumerate(case["camera_ids"])
        if camera_id not in set(args.heldout_camera_ids)
    ]
    if not train_indices:
        parser.error("at least one camera must remain for optimization")
    pcd_path = args.pcd or (
        args.subject_root / "sequences" / args.sequence / "pointclouds" / f"frame_{args.frame:05d}.pcd"
    )
    pcd = read_point_cloud(pcd_path)
    region_masks, region_counts = build_distance_region_masks(
        pcd["xyz"],
        flame_vertices_np,
        case["viewmats"],
        case["Ks"],
        case["alpha"],
        int(case["height"]),
        int(case["width"]),
    )

    base = torch.from_numpy(base_np).to(device)
    quats = quaternions_from_z(torch.from_numpy(normals_np).to(device))
    color_logits = torch.nn.Parameter(torch.from_numpy(logit(colors_np)).to(device))
    scale_initial = np.tile(np.array([0.0030, 0.0030, 0.0005], dtype=np.float32), (len(base_np), 1))
    log_scales = torch.nn.Parameter(torch.from_numpy(np.log(scale_initial)).to(device))
    opacity_logits = torch.nn.Parameter(torch.zeros(len(base_np), device=device))
    raw_delta = torch.nn.Parameter(torch.zeros_like(base))

    parameter_groups = [
        {"params": [color_logits], "lr": 1e-2},
        {"params": [log_scales], "lr": 3e-3},
        {"params": [opacity_logits], "lr": 1e-2},
    ]
    raw_position_lr = (
        args.position_lr / (args.radius_mm / 1000.0) if args.radius_mm > 0 else args.position_lr
    )
    parameter_groups.append({"params": [raw_delta], "lr": raw_position_lr})
    optimizer = torch.optim.Adam(parameter_groups)

    train_index_t = torch.tensor(train_indices, dtype=torch.long, device=device)
    all_targets = torch.from_numpy(case["images"]).to(device)
    all_target_alpha = torch.from_numpy(case["alpha"]).to(device)
    all_head_mask = torch.from_numpy(region_masks["head"].astype(np.float32)).to(device)
    targets = all_targets[train_index_t]
    target_alpha = all_target_alpha[train_index_t]
    head_mask = all_head_mask[train_index_t]
    target_background = target_alpha < 0.05
    all_viewmats = torch.from_numpy(case["viewmats"]).to(device)
    all_intrinsics = torch.from_numpy(case["Ks"]).to(device)
    viewmats = all_viewmats[train_index_t]
    intrinsics = all_intrinsics[train_index_t]
    history = []

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        means = means_for_constraint(base, raw_delta, args.radius_mm)
        scales = torch.exp(log_scales).clamp(5e-5, args.max_scale_mm / 1000.0)
        colors = torch.sigmoid(color_logits)
        opacities = torch.sigmoid(opacity_logits)
        rendered, alpha, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            intrinsics,
            int(case["width"]),
            int(case["height"]),
            packed=True,
        )
        rendered = rendered + (1.0 - alpha)
        rgb_loss = (torch.abs(rendered - targets) * head_mask).sum() / (head_mask.sum() * 3.0).clamp_min(1.0)
        alpha_loss = (torch.abs(alpha - target_alpha) * head_mask).sum() / head_mask.sum().clamp_min(1.0)
        leak_loss = (alpha * target_background).sum() / target_background.sum().clamp_min(1.0)
        scale_penalty = (
            torch.relu(scales - args.scale_penalty_threshold_mm / 1000.0).square().mean()
            * args.scale_penalty_weight
        )
        loss = rgb_loss + 0.2 * alpha_loss + 0.05 * leak_loss + scale_penalty
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            record = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "rgb_l1": float(rgb_loss.detach()),
                "alpha_l1": float(alpha_loss.detach()),
                "leak": float(leak_loss.detach()),
                "mean_scale_mm": float(scales.detach().mean() * 1000.0),
                "mean_offset_mm": float((means.detach() - base).norm(dim=-1).mean() * 1000.0),
            }
            history.append(record)
            print(json.dumps(record))

    with torch.no_grad():
        means = means_for_constraint(base, raw_delta, args.radius_mm)
        scales = torch.exp(log_scales).clamp(5e-5, args.max_scale_mm / 1000.0)
        colors = torch.sigmoid(color_logits)
        opacities = torch.sigmoid(opacity_logits)
        rendered, alpha, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            all_viewmats,
            all_intrinsics,
            int(case["width"]),
            int(case["height"]),
            packed=True,
        )
        rendered = rendered + (1.0 - alpha)
    rendered_np = rendered.cpu().numpy()
    alpha_np = alpha.cpu().numpy()
    means_np = means.cpu().numpy()

    evaluation = {}
    for region_name, masks in region_masks.items():
        per_camera = []
        for index, camera_id in enumerate(case["camera_ids"]):
            metrics = {
                "camera_id": camera_id,
                **masked_image_metrics(rendered_np[index], case["images"][index], masks[index]),
                **alpha_metrics(alpha_np[index] * masks[index], case["alpha"][index] * masks[index]),
            }
            per_camera.append(metrics)
        metric_names = ("masked_mse", "masked_mae", "masked_psnr", "alpha_mae", "alpha_iou")
        evaluation[region_name] = {
            key: float(np.mean([row[key] for row in per_camera])) for key in metric_names
        }
        evaluation[region_name]["train"] = {
            key: float(np.mean([per_camera[index][key] for index in train_indices]))
            for key in metric_names
        }
        evaluation[region_name]["heldout"] = (
            {
                key: float(np.mean([per_camera[index][key] for index in heldout_indices]))
                for key in metric_names
            }
            if heldout_indices
            else None
        )
        evaluation[region_name]["per_camera"] = per_camera

    means_to_pcd, _ = cKDTree(pcd["xyz"]).query(means_np, workers=-1)
    pcd_to_base, _ = cKDTree(flame_vertices_np).query(pcd["xyz"], workers=-1)
    pcd_to_means, _ = cKDTree(means_np).query(pcd["xyz"], workers=-1)
    geometry = {
        "means_to_pcd_mean_mm": float(means_to_pcd.mean() * 1000.0),
        "means_to_pcd_median_mm": float(np.median(means_to_pcd) * 1000.0),
    }
    bands = {
        "supported": pcd_to_base <= 0.0075,
        "transition": (pcd_to_base > 0.0075) & (pcd_to_base <= 0.015),
        "unsupported": (pcd_to_base > 0.015) & (pcd_to_base <= 0.08),
    }
    for name, band in bands.items():
        values = pcd_to_means[band] * 1000.0
        geometry[f"pcd_to_means_{name}_mean_mm"] = float(values.mean())
        geometry[f"pcd_to_means_{name}_median_mm"] = float(np.median(values))

    args.output.mkdir(parents=True, exist_ok=True)
    for index, camera_id in enumerate(case["camera_ids"]):
        save_rgb(args.output / f"render_{camera_id}.png", rendered_np[index])
    pairs = []
    for index in range(min(6, len(case["camera_ids"]))):
        pairs.extend([case["images"][index], rendered_np[index]])
    save_montage(args.output / "target_render_montage.png", pairs, columns=4)
    np.savez_compressed(
        args.output / "model.npz",
        means=means_np.astype(np.float32),
        scales=scales.cpu().numpy().astype(np.float32),
        colors=colors.cpu().numpy().astype(np.float32),
        opacities=opacities.cpu().numpy().astype(np.float32),
    )
    report = {
        "kind": "geometry_constraint_oracle_fit",
        "fit": str(args.fit),
        "pcd": str(pcd_path),
        "radius_mm": args.radius_mm,
        "steps": args.steps,
        "position_lr_m": args.position_lr,
        "raw_position_lr": raw_position_lr,
        "seed": args.seed,
        "gaussian_count": len(base_np),
        "density_scaffold": density_report,
        "total_trainable_parameters": int(sum(parameter.numel() for group in parameter_groups for parameter in group["params"])),
        "active_position_parameters": 0 if args.radius_mm == 0 else int(raw_delta.numel()),
        "camera_ids": case["camera_ids"],
        "train_camera_ids": [case["camera_ids"][index] for index in train_indices],
        "heldout_camera_ids": [case["camera_ids"][index] for index in heldout_indices],
        "resolution": [int(case["height"]), int(case["width"])],
        "alpha_source": case["alpha_source"],
        "background_threshold": case["background_threshold"],
        "scale_constraint": {
            "initial_tangent_mm": 3.0,
            "initial_normal_mm": 0.5,
            "max_axis_mm": args.max_scale_mm,
            "penalty_threshold_mm": args.scale_penalty_threshold_mm,
            "penalty_weight": args.scale_penalty_weight,
        },
        "region_definition": {
            "supported_mm": "<=7.5 nearest fitted FLAME vertex",
            "transition_mm": "(7.5,15]",
            "unsupported_mm": "(15,80]",
        },
        "region_counts": region_counts,
        "history": history,
        "evaluation": evaluation,
        "geometry": geometry,
        "warning": "Single-case oracle pilot; not a population-level finding.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    compact_evaluation = {
        region: {key: value for key, value in metrics.items() if key != "per_camera"}
        for region, metrics in evaluation.items()
    }
    print(
        json.dumps(
            {
                "output": str(args.output),
                "radius_mm": args.radius_mm,
                "evaluation": compact_evaluation,
                "geometry": geometry,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
