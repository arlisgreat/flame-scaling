#!/usr/bin/env python3
"""Re-render a trained hard/adaptive animation comparison at high resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from flame_oracle.flame import (
    apply_dynamic_flame,
    apply_tracking_world_transform,
    load_flame_dynamic_assets,
    load_flame_linear_assets,
)
from flame_oracle.io import read_video_frame
from scripts.build_visual_hull import project
from scripts.run_animation_oracle import (
    DEFAULT_ASSET_ROOT,
    local_means_for_method,
    render_frames,
    tracking_codes,
)


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def metric_for_frame(report: dict[str, object], region: str, frame: int) -> float:
    rows = report["evaluation"][region]["per_frame"]
    return float(next(row["masked_psnr"] for row in rows if int(row["frame"]) == frame))


def crop_bounds(points: np.ndarray, width: int, height: int, pad: int) -> tuple[int, int, int, int]:
    x0 = max(0, int(np.floor(points[:, 0].min())) - pad)
    x1 = min(width, int(np.ceil(points[:, 0].max())) + pad + 1)
    y0 = max(0, int(np.floor(points[:, 1].min())) - pad)
    y1 = min(height, int(np.ceil(points[:, 1].max())) + pad + 1)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, x1 = cx - side // 2, cx + (side + 1) // 2
    y0, y1 = cy - side // 2, cy + (side + 1) // 2
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="486")
    parser.add_argument("--sequence", default="EXP-5-mouth")
    parser.add_argument("--frame", type=int, default=7)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/mono_flame_avatar"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/runs/animation_multisubject_exp5/nall"),
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/analysis/animation_multisubject_exp5/"
            "highres_hard_vs_adaptive_subject486_frame0007.png"
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    subject_root = args.data_root / args.subject
    sequence_root = subject_root / "sequences" / args.sequence
    tracking = np.load(sequence_root / "tracking" / "flame2023_tracking_v2.npz")
    camera_id = "222200037"
    video_path = sequence_root / "images" / f"cam_{camera_id}.mp4"
    alpha_path = sequence_root / "alpha_maps" / f"cam_{camera_id}.mp4"
    target = read_video_frame(video_path, args.frame, args.height).astype(np.float32) / 255.0
    target_alpha = read_video_frame(alpha_path, args.frame, args.height)[..., 0].astype(np.float32) / 255.0
    height, width = target.shape[:2]

    calibration = json.loads((subject_root / "calibration" / "camera_params.json").read_text())
    capture = cv2.VideoCapture(str(video_path))
    source_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    intrinsic = np.asarray(calibration["intrinsics"][camera_id], np.float32)
    intrinsic[:2] *= args.height / source_height
    viewmat = np.asarray(calibration["world_2_cam"][camera_id], np.float32)

    dynamic_np = load_flame_dynamic_assets(args.asset_root / "flame2023.pkl", 300, 100)
    linear_np = load_flame_linear_assets(
        args.asset_root / "flame2023.pkl", args.asset_root / "FLAME_masks.pkl", 300, 100
    )
    assets = {
        key: torch.from_numpy(value).to(device)
        for key, value in dynamic_np.items()
        if isinstance(value, np.ndarray) and key != "faces"
    }
    with torch.no_grad():
        shape = torch.from_numpy(tracking["shape"]).float().to(device)
        expression = torch.from_numpy(tracking["expression"]).float().to(device)
        neck = torch.from_numpy(tracking["neck"]).float().to(device)
        jaw = torch.from_numpy(tracking["jaw"]).float().to(device)
        eyes = torch.from_numpy(tracking["eyes"]).float().to(device)
        hard_local = apply_dynamic_flame(assets, shape, expression, neck, jaw, eyes)
        neutral_local = apply_dynamic_flame(
            assets,
            shape,
            torch.zeros_like(expression[:1]),
            torch.zeros_like(neck[:1]),
            torch.zeros_like(jaw[:1]),
            torch.zeros_like(eyes[:1]),
        )[0]
        rotations = torch.from_numpy(tracking["rotation_matrices"]).float().to(device)
        translations = torch.from_numpy(tracking["translation"]).float().to(device)
        scale = torch.from_numpy(tracking["scale"]).float().to(device)
        hard_world = apply_tracking_world_transform(hard_local, rotations, translations, scale)

    codes_np, _ = tracking_codes(tracking, 16)
    codes = torch.from_numpy(codes_np[[args.frame]]).to(device)
    viewmat_t = torch.from_numpy(viewmat).to(device)
    intrinsic_t = torch.from_numpy(intrinsic).to(device)
    quaternions = torch.zeros((hard_local.shape[1], 4), device=device)
    quaternions[:, 0] = 1.0

    renders: dict[str, np.ndarray] = {}
    oral_psnr: dict[str, float] = {}
    for method in ("hard", "adaptive"):
        run = args.run_root / args.subject / method
        with np.load(run / "model.npz") as saved:
            deformation_basis = torch.from_numpy(saved["deformation_basis"]).to(device)
            release_logits = torch.from_numpy(logit(saved["release_gates"])).to(device)
            scales = torch.from_numpy(saved["scales"]).to(device)
            colors = torch.from_numpy(saved["colors"]).to(device)
            opacities = torch.from_numpy(saved["opacities"]).to(device)
        with torch.no_grad():
            local, _, _ = local_means_for_method(
                method,
                hard_local[[args.frame]],
                neutral_local,
                codes,
                deformation_basis,
                release_logits,
                30.0,
            )
            world = apply_tracking_world_transform(
                local, rotations[[args.frame]], translations[[args.frame]], scale
            )
            rendered, _ = render_frames(
                world,
                quaternions,
                scales,
                opacities,
                colors,
                viewmat_t,
                intrinsic_t,
                width,
                height,
            )
        renders[method] = rendered[0].cpu().numpy()
        report = json.loads((run / "metrics.json").read_text())
        oral_psnr[method] = metric_for_frame(report, "oral", args.frame)

    lips = linear_np["masks"]["lips"]
    u, v, z = project(hard_world[args.frame].cpu().numpy()[lips], viewmat, intrinsic)
    valid = (z > 0) & np.isfinite(u) & np.isfinite(v)
    mouth_points = np.stack([u[valid], v[valid]], axis=1)
    mouth_crop = crop_bounds(mouth_points, width, height, pad=round(height * 0.045))

    images = [target, renders["hard"], renders["adaptive"]]
    titles = [
        "Target",
        f"Hard FLAME\noral PSNR {oral_psnr['hard']:.2f} dB",
        f"Adaptive release (ours)\noral PSNR {oral_psnr['adaptive']:.2f} dB",
    ]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.8, 7.2),
        gridspec_kw={"height_ratios": [1.0, 0.72]},
        facecolor="white",
    )
    for column, (image, title) in enumerate(zip(images, titles)):
        axes[0, column].imshow(np.clip(image, 0, 1))
        axes[0, column].set_title(title, fontsize=14, fontweight="bold", pad=9)
        x0, y0, x1, y1 = mouth_crop
        axes[0, column].add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ff8c42", linewidth=2)
        )
        axes[1, column].imshow(np.clip(image[y0:y1, x0:x1], 0, 1), interpolation="nearest")
        axes[1, column].set_title("Oral crop", fontsize=11, pad=5)
        for axis in axes[:, column]:
            axis.axis("off")
    fig.tight_layout(pad=1.3, h_pad=0.8, w_pad=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
