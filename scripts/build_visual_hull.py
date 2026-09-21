#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from flame_oracle.io import load_multiview_frame, save_montage, save_rgb


def project(points: np.ndarray, viewmat: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = points @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = camera[:, 2]
    u = K[0, 0] * camera[:, 0] / np.maximum(z, 1e-8) + K[0, 2]
    v = K[1, 1] * camera[:, 1] / np.maximum(z, 1e-8) + K[1, 2]
    return u, v, z


def carve_grid(
    xyz: np.ndarray,
    masks: np.ndarray,
    viewmats: np.ndarray,
    intrinsics: np.ndarray,
    minimum_votes: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = masks.shape[1:]
    occupancy = np.zeros(len(xyz), dtype=bool)
    vote_histogram = np.zeros(len(viewmats) + 1, dtype=np.int64)
    hard_masks = masks > 0.5
    for start in range(0, len(xyz), chunk_size):
        stop = min(start + chunk_size, len(xyz))
        points = xyz[start:stop]
        votes = np.zeros(len(points), dtype=np.uint8)
        for mask, viewmat, K in zip(hard_masks, viewmats, intrinsics):
            u, v, z = project(points, viewmat, K)
            inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            ids = np.flatnonzero(inside)
            if len(ids):
                votes[ids] += mask[v[ids].astype(np.int32), u[ids].astype(np.int32)]
        occupancy[start:stop] = votes >= minimum_votes
        vote_histogram += np.bincount(votes, minlength=len(viewmats) + 1)
    return occupancy, vote_histogram


def surface_colors(
    xyz: np.ndarray,
    images: np.ndarray,
    alpha: np.ndarray,
    viewmats: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    height, width = images.shape[1:3]
    samples = []
    for image, mask, viewmat, K in zip(images, alpha[..., 0], viewmats, intrinsics):
        u, v, z = project(xyz, viewmat, K)
        inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        ui = np.clip(u.astype(np.int32), 0, width - 1)
        vi = np.clip(v.astype(np.int32), 0, height - 1)
        valid = inside & (mask[vi, ui] > 0.5)
        values = np.full((len(xyz), 3), np.nan, dtype=np.float32)
        values[valid] = image[vi[valid], ui[valid]]
        samples.append(values)
    with np.errstate(all="ignore"):
        colors = np.nanmedian(np.stack(samples), axis=0)
    colors = np.nan_to_num(colors, nan=0.5)
    return np.clip(colors * 255.0, 0, 255).astype(np.uint8)


def projection_overlay(
    image: np.ndarray,
    points: np.ndarray,
    viewmat: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]
    u, v, z = project(points, viewmat, K)
    valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid].astype(np.int32)
    v = v[valid].astype(np.int32)
    z = z[valid]
    order = np.argsort(z)[::-1]
    coverage = np.zeros((height, width), dtype=np.uint8)
    coverage[v[order], u[order]] = 255
    coverage = cv2.dilate(coverage, np.ones((3, 3), np.uint8), iterations=1)
    overlay = image.copy()
    overlay[coverage > 0] = 0.55 * overlay[coverage > 0] + 0.45 * np.array([1.0, 0.15, 0.05])
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--background-threshold", type=int, default=30)
    parser.add_argument("--voxel-mm", type=float, default=2.5)
    parser.add_argument("--minimum-votes", type=int, default=14)
    parser.add_argument("--x-range", type=float, nargs=2, default=(-0.18, 0.18))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-0.22, 0.16))
    parser.add_argument("--z-range", type=float, nargs=2, default=(-0.10, 0.18))
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case = load_multiview_frame(
        args.subject_root,
        args.sequence,
        args.frame,
        args.height,
        background_threshold=args.background_threshold,
    )
    step = args.voxel_mm / 1000.0
    axes = [
        np.arange(bounds[0], bounds[1] + step * 0.5, step, dtype=np.float32)
        for bounds in (args.x_range, args.y_range, args.z_range)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    xyz = grid.reshape(-1, 3)
    flat_occupancy, vote_histogram = carve_grid(
        xyz,
        case["alpha"][..., 0],
        case["viewmats"],
        case["Ks"],
        args.minimum_votes,
        args.chunk_size,
    )
    occupancy = flat_occupancy.reshape(grid.shape[:-1])
    occupancy = ndimage.binary_closing(occupancy, iterations=1)
    occupancy = ndimage.binary_fill_holes(occupancy)
    labels, component_count = ndimage.label(occupancy)
    if component_count == 0:
        raise RuntimeError("visual-hull carving produced no occupied voxels")
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    occupancy = labels == int(component_sizes.argmax())
    surface = occupancy & ~ndimage.binary_erosion(occupancy, iterations=1)
    surface_xyz = grid[surface]

    smooth = ndimage.gaussian_filter(occupancy.astype(np.float32), sigma=1.0)
    gradients = np.stack(np.gradient(smooth, step, edge_order=1), axis=-1)
    normals = -gradients[surface]
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = surface_xyz - surface_xyz.mean(axis=0, keepdims=True)
    fallback /= np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-8)
    normals = np.where(normal_norm > 1e-6, normals / np.maximum(normal_norm, 1e-8), fallback)
    rgb = surface_colors(
        surface_xyz,
        case["images"],
        case["alpha"],
        case["viewmats"],
        case["Ks"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        xyz=surface_xyz.astype(np.float32),
        normals=normals.astype(np.float32),
        rgb=rgb,
        occupancy_shape=np.asarray(occupancy.shape, dtype=np.int32),
        occupancy_origin=np.asarray([axis[0] for axis in axes], dtype=np.float32),
        voxel_size_m=np.asarray([step], dtype=np.float32),
    )
    masks = []
    overlays = []
    for image, alpha, viewmat, K in zip(
        case["images"], case["alpha"], case["viewmats"], case["Ks"]
    ):
        masks.append(np.repeat(alpha, 3, axis=-1))
        overlays.append(projection_overlay(image, surface_xyz, viewmat, K))
    save_montage(args.output.with_name(args.output.stem + "_masks.png"), masks, columns=4)
    save_montage(args.output.with_name(args.output.stem + "_overlays.png"), overlays, columns=4)

    report = {
        "kind": "silhouette_visual_hull_proxy",
        "subject_root": str(args.subject_root),
        "sequence": args.sequence,
        "frame": args.frame,
        "camera_ids": case["camera_ids"],
        "alpha_source": case["alpha_source"],
        "background_threshold": args.background_threshold,
        "voxel_mm": args.voxel_mm,
        "minimum_votes": args.minimum_votes,
        "bounds_m": {"x": args.x_range, "y": args.y_range, "z": args.z_range},
        "grid_shape": list(occupancy.shape),
        "grid_voxels": int(occupancy.size),
        "occupied_voxels": int(occupancy.sum()),
        "surface_points": int(len(surface_xyz)),
        "pre_component_count": int(component_count),
        "vote_histogram": vote_histogram.tolist(),
        "warning": (
            "Silhouette visual hull is a conservative geometric proxy, not laser-scanned ground truth; "
            "concavities such as the mouth cavity are not identifiable."
        ),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
