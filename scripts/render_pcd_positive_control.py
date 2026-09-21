#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization

from flame_oracle.io import load_multiview_frame, read_binary_pcd, save_montage, save_rgb
from flame_oracle.metrics import alpha_metrics, masked_image_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject-root",
        type=Path,
        default=Path("/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs/388"),
    )
    parser.add_argument("--sequence", default="GLASSES")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--scales-mm", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--normal-scale-mm", type=float, default=0.08)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case = load_multiview_frame(args.subject_root, args.sequence, args.frame, args.height)
    pcd = read_binary_pcd(
        args.subject_root / "sequences" / args.sequence / "pointclouds" / f"frame_{args.frame:05d}.pcd"
    )
    rng = np.random.default_rng(20260715)
    count = min(args.max_points, len(pcd["xyz"]))
    indices = rng.choice(len(pcd["xyz"]), count, replace=False)

    device = torch.device("cuda")
    means = torch.from_numpy(pcd["xyz"][indices]).to(device)
    colors = torch.from_numpy(pcd["rgb"][indices].astype(np.float32) / 255.0).to(device)
    normals = torch.from_numpy(pcd["normals"][indices]).to(device)
    normals = torch.nn.functional.normalize(normals, dim=-1)
    # Scalar-first quaternion rotating the local +Z axis onto the PCD normal.
    quats = torch.stack(
        [1.0 + normals[:, 2], -normals[:, 1], normals[:, 0], torch.zeros_like(normals[:, 0])],
        dim=-1,
    )
    opposite = quats.norm(dim=-1) < 1e-6
    quats[opposite] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    opacities = torch.full((count,), 0.95, device=device)
    viewmats = torch.from_numpy(case["viewmats"]).to(device)
    Ks = torch.from_numpy(case["Ks"]).to(device)
    targets = case["images"]
    target_alpha = case["alpha"]

    reports = []
    best = None
    with torch.no_grad():
        for scale_mm in args.scales_mm:
            scales = torch.empty((count, 3), device=device)
            scales[:, :2] = scale_mm / 1000.0
            scales[:, 2] = args.normal_scale_mm / 1000.0
            rendered, alpha, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                Ks,
                int(case["width"]),
                int(case["height"]),
                packed=True,
            )
            rendered = rendered + (1.0 - alpha)
            rendered_np = rendered.cpu().numpy()
            alpha_np = alpha.cpu().numpy()
            per_camera = []
            for index, camera_id in enumerate(case["camera_ids"]):
                metrics = {
                    "camera_id": camera_id,
                    **masked_image_metrics(rendered_np[index], targets[index], target_alpha[index] > 0.5),
                    **alpha_metrics(alpha_np[index], target_alpha[index]),
                }
                per_camera.append(metrics)
            aggregate = {
                key: float(np.mean([row[key] for row in per_camera]))
                for key in ("masked_mse", "masked_mae", "masked_psnr", "alpha_mae", "alpha_iou")
            }
            record = {"scale_mm": scale_mm, "aggregate": aggregate, "per_camera": per_camera}
            reports.append(record)
            if best is None or aggregate["masked_mse"] < best[0]:
                best = (aggregate["masked_mse"], scale_mm, rendered_np, alpha_np)

    assert best is not None
    _, best_scale, best_rendered, best_alpha = best
    args.output.mkdir(parents=True, exist_ok=True)
    for index, camera_id in enumerate(case["camera_ids"]):
        save_rgb(args.output / f"render_{camera_id}.png", best_rendered[index])
        save_rgb(args.output / f"target_{camera_id}.png", targets[index])
    pairs = []
    for index in range(min(4, len(case["camera_ids"]))):
        pairs.extend([targets[index], best_rendered[index]])
    save_montage(args.output / "target_render_montage.png", pairs, columns=4)

    report = {
        "kind": "gt_pointcloud_camera_renderer_positive_control",
        "subject_root": str(args.subject_root),
        "sequence": args.sequence,
        "frame": args.frame,
        "resolution": [int(case["height"]), int(case["width"])],
        "camera_ids": case["camera_ids"],
        "source_point_count": len(pcd["xyz"]),
        "rendered_point_count": count,
        "best_scale_mm_by_masked_mse": best_scale,
        "normal_scale_mm": args.normal_scale_mm,
        "sweep": reports,
        "interpretation": "A plumbing positive control only; it is not a FLAME ceiling result.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "best_scale_mm": best_scale, "aggregate": next(r["aggregate"] for r in reports if r["scale_mm"] == best_scale)}, indent=2))


if __name__ == "__main__":
    main()
