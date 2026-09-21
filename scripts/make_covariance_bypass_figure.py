#!/usr/bin/env python3
"""Make a paper-ready visual of the FLAME covariance-bypass failure mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from gsplat import rasterization

from flame_oracle.io import load_multiview_frame
from flame_oracle.render import quaternions_from_z, vertex_normals


def project(points: np.ndarray, viewmat: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = points @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = camera[:, 2]
    pixels = np.empty((len(points), 2), dtype=np.float32)
    pixels[:, 0] = K[0, 0] * camera[:, 0] / z + K[0, 2]
    pixels[:, 1] = K[1, 1] * camera[:, 1] / z + K[1, 2]
    return pixels, camera


def quaternion_matrices_wxyz(quaternions: np.ndarray) -> np.ndarray:
    q = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True).clip(min=1e-8)
    w, x, y, z = q.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def projected_covariances(
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    viewmat: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pixels, camera = project(means, viewmat, K)
    world_cov = rotations @ np.vectorize(np.diag, signature="(n)->(n,n)")(scales**2) @ np.transpose(rotations, (0, 2, 1))
    camera_cov = viewmat[:3, :3][None] @ world_cov @ viewmat[:3, :3].T[None]
    x, y, z = camera.T
    jacobian = np.zeros((len(means), 2, 3), dtype=np.float32)
    jacobian[:, 0, 0] = K[0, 0] / z
    jacobian[:, 0, 2] = -K[0, 0] * x / (z * z)
    jacobian[:, 1, 1] = K[1, 1] / z
    jacobian[:, 1, 2] = -K[1, 1] * y / (z * z)
    covariance_2d = jacobian @ camera_cov @ np.transpose(jacobian, (0, 2, 1))
    return pixels, covariance_2d


def render_high_resolution(
    model: dict[str, np.ndarray],
    normals: np.ndarray,
    viewmat: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    device = torch.device("cuda")
    with torch.no_grad():
        quaternions = quaternions_from_z(torch.from_numpy(normals).to(device))
        rendered, alpha, _ = rasterization(
            torch.from_numpy(model["means"]).to(device),
            quaternions,
            torch.from_numpy(model["scales"]).to(device),
            torch.from_numpy(model["opacities"]).to(device),
            torch.from_numpy(model["colors"]).to(device),
            torch.from_numpy(viewmat[None]).to(device),
            torch.from_numpy(K[None]).to(device),
            width,
            height,
            packed=True,
        )
        rendered = rendered + (1.0 - alpha)
    return rendered[0].cpu().numpy()


def crop_from_alpha(alpha: np.ndarray, pad_fraction: float = 0.045) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(alpha[..., 0] > 0.05)
    if not len(xs):
        return 0, 0, alpha.shape[1], alpha.shape[0]
    pad = round(max(alpha.shape[:2]) * pad_fraction)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(alpha.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(alpha.shape[0], int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="443")
    parser.add_argument("--sequence", default="FREE")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera", default="222200041")
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("artifacts/runs/multicase_landmark_mean_extent4000/443/hard_cap030"),
    )
    parser.add_argument(
        "--fit",
        type=Path,
        default=Path("artifacts/cache/flame_fit_443_landmark_anchored.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/multicase_landmark_mean_extent4000/covariance_bypass_hair.png"),
    )
    args = parser.parse_args()

    subject_root = Path("/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs") / args.subject
    case = load_multiview_frame(subject_root, args.sequence, args.frame, args.height)
    if args.camera not in case["camera_ids"]:
        raise ValueError(f"camera {args.camera} not found")
    camera_index = case["camera_ids"].index(args.camera)
    target = case["images"][camera_index]
    target_alpha = case["alpha"][camera_index]
    viewmat = case["viewmats"][camera_index]
    K = case["Ks"][camera_index]

    fit = np.load(args.fit)
    vertices = fit["vertices"].astype(np.float32)
    faces = fit["faces"].astype(np.int64)
    normals = vertex_normals(vertices, faces)
    with np.load(args.run / "model.npz") as payload:
        model = {key: payload[key].astype(np.float32) for key in payload.files}
    metrics = json.loads((args.run / "metrics.json").read_text())
    if not np.allclose(model["means"], vertices, atol=1e-7):
        raise ValueError("the requested visualization assumes hard means fixed to FLAME vertices")

    rendered = render_high_resolution(
        model,
        normals,
        viewmat,
        K,
        int(case["width"]),
        int(case["height"]),
    )

    centers_2d, camera_points = project(model["means"], viewmat, K)
    quaternion_np = quaternions_from_z(torch.from_numpy(normals)).numpy()
    rotations = quaternion_matrices_wxyz(quaternion_np)
    _, covariance_2d = projected_covariances(model["means"], model["scales"], rotations, viewmat, K)

    x0, y0, x1, y1 = crop_from_alpha(target_alpha)
    crop_width, crop_height = x1 - x0, y1 - y0
    visible = (
        (camera_points[:, 2] > 0)
        & (centers_2d[:, 0] >= x0)
        & (centers_2d[:, 0] < x1)
        & (centers_2d[:, 1] >= y0)
        & (centers_2d[:, 1] < y1)
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_2d)
    eigenvalues = np.clip(eigenvalues, 0, None)
    major_sigma_px = np.sqrt(eigenvalues[:, 1])
    axis_max_mm = model["scales"].max(axis=1) * 1000.0
    candidate = visible & (axis_max_mm >= np.quantile(axis_max_mm, 0.90)) & (model["opacities"] >= 0.75)
    ranked = np.where(candidate)[0]
    ranked = ranked[np.argsort(major_sigma_px[ranked] * model["opacities"][ranked])[::-1]]

    selected: list[int] = []
    minimum_spacing = 0.055 * max(crop_width, crop_height)
    for index in ranked:
        if all(np.linalg.norm(centers_2d[index] - centers_2d[previous]) >= minimum_spacing for previous in selected):
            selected.append(int(index))
        if len(selected) == 22:
            break

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    line_visible = visible[edges[:, 0]] & visible[edges[:, 1]]
    segments = centers_2d[edges[line_visible]]

    faded = 0.78 * np.ones_like(target) + 0.22 * target
    fig = plt.figure(figsize=(13.2, 6.8), facecolor="#f6f7f9")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], left=0.035, right=0.975, top=0.83, bottom=0.15, wspace=0.055)
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1])

    left.imshow(np.clip(faded, 0, 1))
    left.add_collection(LineCollection(segments, colors="#13a7b8", linewidths=0.42, alpha=0.30))
    left.scatter(
        centers_2d[visible, 0],
        centers_2d[visible, 1],
        s=2.1,
        c="#d62728",
        alpha=0.56,
        linewidths=0,
        zorder=3,
    )
    for index in selected:
        values = eigenvalues[index]
        vector = eigenvectors[index, :, 1]
        angle = np.degrees(np.arctan2(vector[1], vector[0]))
        ellipse = Ellipse(
            centers_2d[index],
            width=6.0 * np.sqrt(values[1]),
            height=6.0 * np.sqrt(values[0]),
            angle=angle,
            facecolor="#f4a261",
            edgecolor="#dc6b19",
            linewidth=1.15,
            alpha=0.20,
            zorder=2,
        )
        left.add_patch(ellipse)

    right.imshow(np.clip(rendered, 0, 1))
    for axis in (left, right):
        axis.set_xlim(x0, x1)
        axis.set_ylim(y1, y0)
        axis.set_aspect("equal")
        axis.axis("off")

    left.set_title("Geometry support", fontsize=17, fontweight="bold", pad=12, color="#17202a")
    right.set_title("Rendered appearance", fontsize=17, fontweight="bold", pad=12, color="#17202a")
    left.text(
        0.03,
        0.965,
        "5,023 centers remain on FLAME",
        transform=left.transAxes,
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#9d1b1b",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="white", edgecolor="#e4a7a7", alpha=0.94),
        zorder=10,
    )
    right.text(
        0.97,
        0.965,
        "Hair appears in RGB",
        transform=right.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#8b4a14",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="white", edgecolor="#e5b17e", alpha=0.94),
    )

    legend = [
        Line2D([0], [0], color="#13a7b8", lw=2, label="FLAME mesh"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markersize=6, label="Gaussian center"),
        Line2D([0], [0], color="#dc6b19", lw=5, alpha=0.45, label="selected projected 3σ footprint"),
    ]
    left.legend(handles=legend, loc="lower left", frameon=True, framealpha=0.94, facecolor="white", edgecolor="#d6d9dd", fontsize=9.5)

    unsupported_error = metrics["geometry"]["pcd_to_means_unsupported_mean_mm"]
    p99 = float(np.quantile(axis_max_mm, 0.99))
    fig.suptitle(
        "Covariance bypass: hair pixels without hair-centered Gaussians",
        x=0.505,
        y=0.955,
        fontsize=22,
        fontweight="bold",
        color="#111820",
    )
    fig.text(
        0.505,
        0.895,
        f"Hard FLAME means (0 position DoF)  •  unsupported-region center error {unsupported_error:.1f} mm  •  σmax p99 {p99:.0f} mm",
        ha="center",
        fontsize=11.5,
        color="#4b5661",
    )
    fig.text(
        0.505,
        0.065,
        "The appearance expands into hair; the Gaussian centers do not.",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color="#a04712",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#fff3e8", edgecolor="#f2c399"),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor=fig.get_facecolor())
    fig.savefig(args.output.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
