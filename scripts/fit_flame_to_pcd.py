#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from flame_oracle.flame import apply_linear_flame, load_flame_linear_assets
from flame_oracle.io import read_point_cloud


DEFAULT_ASSET_ROOT = Path("/data1/workspace/airulan/3dv/external/LAM/model_zoo/human_parametric_models/flame_vhap")


def region_metrics(vertices: np.ndarray, xyz: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, object]:
    target_tree = cKDTree(xyz)
    distances, nearest = target_tree.query(vertices, workers=-1)
    metrics = {}
    for region in ("face", "eye_region", "lips", "nose", "left_ear", "right_ear", "scalp", "neck"):
        ids = masks[region]
        values = distances[ids] * 1000.0
        metrics[region] = {
            "count": int(len(ids)),
            "mean_mm": float(np.mean(values)),
            "median_mm": float(np.median(values)),
            "p95_mm": float(np.percentile(values, 95)),
        }
    return {"by_flame_region": metrics, "nearest_target_index": nearest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pcd",
        type=Path,
        default=Path(
            "/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs/388/sequences/GLASSES/pointclouds/frame_00000.pcd"
        ),
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--shape-dims", type=int, default=100)
    parser.add_argument("--expression-dims", type=int, default=50)
    parser.add_argument("--outer-iterations", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=200)
    parser.add_argument("--target-sample", type=int, default=200000)
    parser.add_argument(
        "--target-y-min",
        type=float,
        help="Optional world-space lower y bound used only for scaffold fitting.",
    )
    parser.add_argument("--max-correspondence-mm", type=float, default=25.0)
    parser.add_argument(
        "--exclude-mouth-radius-mm",
        type=float,
        default=0.0,
        help="Exclude template vertices within this distance of FLAME lip vertices from ICP fitting.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shape-prior", type=float, default=1e-3)
    parser.add_argument("--expression-prior", type=float, default=1e-3)
    parser.add_argument(
        "--coefficient-limit",
        type=float,
        default=3.0,
        help="Clamp PCA coefficients to this many nominal standard deviations; <=0 disables.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pcd = read_point_cloud(args.pcd)
    assets = load_flame_linear_assets(
        args.asset_root / "flame2023.pkl",
        args.asset_root / "FLAME_masks.pkl",
        args.shape_dims,
        args.expression_dims,
    )
    template = assets["template"]
    directions = assets["directions"]
    masks = assets["masks"]
    fit_ids = np.unique(
        np.concatenate([masks["face"], masks["left_ear"], masks["right_ear"], masks["scalp"]])
    )
    mouth_excluded = np.zeros(len(template), dtype=bool)
    if args.exclude_mouth_radius_mm > 0:
        distance_to_lips, _ = cKDTree(template[np.asarray(masks["lips"], dtype=np.int64)]).query(
            template,
            workers=-1,
        )
        mouth_excluded = distance_to_lips <= args.exclude_mouth_radius_mm / 1000.0
        fit_ids = fit_ids[~mouth_excluded[fit_ids]]

    rng = np.random.default_rng(args.seed)
    target_pool = pcd["xyz"]
    if args.target_y_min is not None:
        target_pool = target_pool[target_pool[:, 1] >= args.target_y_min]
    if len(target_pool) < 1000:
        raise RuntimeError(f"target crop left too few points: {len(target_pool)}")
    target_indices = rng.choice(len(target_pool), min(args.target_sample, len(target_pool)), replace=False)
    target = target_pool[target_indices]
    target_tree = cKDTree(target)

    # Correct semantic orientation is fixed from the camera-overlay audit. We
    # intentionally do not search flips because an incorrect flip obtains a
    # lower one-sided ICP cost by collapsing onto hair.
    initial = np.eye(4)
    initial[:3, :3] *= 0.95
    initial[:3, 3] = target.mean(axis=0) - (template[fit_ids] @ initial[:3, :3].T).mean(axis=0)
    matrix, _, icp_cost = trimesh.registration.icp(
        template[fit_ids],
        target,
        initial=initial,
        threshold=1e-10,
        max_iterations=100,
        scale=False,
        reflection=False,
    )
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    rotation = matrix[:3, :3] / scale
    rotvec_initial = Rotation.from_matrix(rotation).as_rotvec().astype(np.float32)

    device = torch.device("cuda")
    template_t = torch.from_numpy(template).to(device)
    directions_t = torch.from_numpy(directions).to(device)
    coefficients = torch.nn.Parameter(torch.zeros(directions.shape[-1], device=device))
    rotvec = torch.nn.Parameter(torch.from_numpy(rotvec_initial).to(device))
    log_scale = torch.nn.Parameter(torch.tensor(math.log(scale), dtype=torch.float32, device=device))
    translation = torch.nn.Parameter(torch.from_numpy(matrix[:3, 3].astype(np.float32)).to(device))
    optimizer = torch.optim.Adam(
        [
            {"params": [coefficients], "lr": 2e-2},
            {"params": [rotvec], "lr": 2e-4},
            {"params": [log_scale], "lr": 2e-4},
            {"params": [translation], "lr": 2e-4},
        ]
    )

    fit_ids_t = torch.from_numpy(fit_ids).long().to(device)
    history = []
    max_distance = args.max_correspondence_mm / 1000.0
    for outer in range(args.outer_iterations):
        with torch.no_grad():
            current = apply_linear_flame(template_t, directions_t, coefficients, rotvec, log_scale, translation)
            current_fit = current[fit_ids_t].cpu().numpy()
        correspondence_distance, correspondence_index = target_tree.query(current_fit, workers=-1)
        keep = correspondence_distance < max_distance
        if keep.sum() < 500:
            raise RuntimeError(f"too few correspondences at outer={outer}: {keep.sum()}")
        kept_ids = torch.from_numpy(fit_ids[keep]).long().to(device)
        matched = torch.from_numpy(target[correspondence_index[keep]].astype(np.float32)).to(device)

        for _ in range(args.inner_steps):
            optimizer.zero_grad(set_to_none=True)
            current = apply_linear_flame(template_t, directions_t, coefficients, rotvec, log_scale, translation)
            residual = current[kept_ids] - matched
            data_loss = torch.nn.functional.smooth_l1_loss(
                residual,
                torch.zeros_like(residual),
                beta=0.003,
            )
            shape = coefficients[: args.shape_dims]
            expression = coefficients[args.shape_dims :]
            expression_regularizer = (
                args.expression_prior * expression.square().mean()
                if expression.numel()
                else torch.zeros((), device=device)
            )
            regularizer = args.shape_prior * shape.square().mean() + expression_regularizer
            scale_prior = 1e-3 * (log_scale - math.log(0.95)).square()
            loss = data_loss + regularizer + scale_prior
            loss.backward()
            optimizer.step()
            if args.coefficient_limit > 0:
                with torch.no_grad():
                    coefficients.clamp_(-args.coefficient_limit, args.coefficient_limit)

        with torch.no_grad():
            current = apply_linear_flame(template_t, directions_t, coefficients, rotvec, log_scale, translation)
            post_distance, _ = target_tree.query(current[fit_ids_t].cpu().numpy(), workers=-1)
        record = {
            "outer": outer,
            "kept_correspondences": int(keep.sum()),
            "mean_mm": float(post_distance.mean() * 1000.0),
            "median_mm": float(np.median(post_distance) * 1000.0),
            "p95_mm": float(np.percentile(post_distance, 95) * 1000.0),
        }
        history.append(record)
        print(json.dumps(record))

    with torch.no_grad():
        vertices = apply_linear_flame(template_t, directions_t, coefficients, rotvec, log_scale, translation).cpu().numpy()
    target_tree_full = cKDTree(pcd["xyz"])
    nearest_distance, nearest_index = target_tree_full.query(vertices, workers=-1)
    initial_colors = pcd["rgb"][nearest_index]
    region = region_metrics(vertices, pcd["xyz"], masks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        vertices=vertices.astype(np.float32),
        faces=assets["faces"].astype(np.int64),
        coefficients=coefficients.detach().cpu().numpy(),
        rotvec=rotvec.detach().cpu().numpy(),
        scale=np.array([float(torch.exp(log_scale).detach().cpu())], dtype=np.float32),
        translation=translation.detach().cpu().numpy(),
        initial_colors=initial_colors,
        nearest_pcd_distance=nearest_distance.astype(np.float32),
    )
    report = {
        "kind": "linear_flame_to_gt_pointcloud_fit",
        "pcd": str(args.pcd),
        "output": str(args.output),
        "icp_cost": float(icp_cost),
        "semantic_orientation": "no_flip_fixed_by_camera_overlay",
        "shape_dimensions": args.shape_dims,
        "expression_dimensions": args.expression_dims,
        "target_y_min": args.target_y_min,
        "target_pool_points": int(len(target_pool)),
        "exclude_mouth_radius_mm": args.exclude_mouth_radius_mm,
        "excluded_mouth_vertices": int(mouth_excluded.sum()),
        "fit_vertex_count": int(len(fit_ids)),
        "shape_prior": args.shape_prior,
        "expression_prior": args.expression_prior,
        "coefficient_limit": args.coefficient_limit,
        "coefficient_statistics": {
            "l2": float(coefficients.detach().norm().cpu()),
            "mean_abs": float(coefficients.detach().abs().mean().cpu()),
            "max_abs": float(coefficients.detach().abs().max().cpu()),
        },
        "history": history,
        "final": {key: value for key, value in region.items() if key != "nearest_target_index"},
        "warning": "This is an oracle scaffold fit, not an avatar reconstruction result.",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "report": str(report_path), "final": report["final"]}, indent=2))


if __name__ == "__main__":
    main()
