from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree


def build_distance_region_masks(
    pcd_xyz: np.ndarray,
    flame_vertices: np.ndarray,
    viewmats: np.ndarray,
    intrinsics: np.ndarray,
    target_alpha: np.ndarray,
    height: int,
    width: int,
    supported_mm: float = 7.5,
    unsupported_mm: float = 15.0,
    head_max_mm: float = 80.0,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Project FLAME-distance bands from the GT point cloud into each camera."""
    distances, _ = cKDTree(flame_vertices).query(pcd_xyz, workers=-1)
    distance_mm = distances * 1000.0
    labels = np.zeros(len(pcd_xyz), dtype=np.uint8)
    labels[distance_mm <= supported_mm] = 1
    labels[(distance_mm > supported_mm) & (distance_mm <= unsupported_mm)] = 2
    labels[(distance_mm > unsupported_mm) & (distance_mm <= head_max_mm)] = 3

    maps = []
    for viewmat, K in zip(viewmats, intrinsics):
        camera = pcd_xyz @ viewmat[:3, :3].T + viewmat[:3, 3]
        valid = (camera[:, 2] > 0) & (labels > 0)
        camera = camera[valid]
        visible_labels = labels[valid]
        u = K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2]
        v = K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u = u[valid].astype(np.int32)
        v = v[valid].astype(np.int32)
        z = camera[valid, 2]
        visible_labels = visible_labels[valid]
        order = np.argsort(z)[::-1]
        label_map = np.zeros((height, width), dtype=np.uint8)
        label_map[v[order], u[order]] = visible_labels[order]
        maps.append(label_map)

    maps = np.stack(maps)
    alpha = target_alpha[..., 0] > 0.5
    kernel = np.ones((3, 3), dtype=np.uint8)
    result = {}
    for name, label in (("supported", 1), ("transition", 2), ("unsupported", 3)):
        dense = np.stack([cv2.dilate((item == label).astype(np.uint8), kernel, iterations=1) for item in maps])
        result[name] = (dense.astype(bool) & alpha)[..., None]
    # Resolve dilation overlap in favor of the more constrained region.
    result["transition"] &= ~result["supported"]
    result["unsupported"] &= ~(result["supported"] | result["transition"])
    result["head"] = result["supported"] | result["transition"] | result["unsupported"]
    counts = {
        "pcd_supported": int((labels == 1).sum()),
        "pcd_transition": int((labels == 2).sum()),
        "pcd_unsupported": int((labels == 3).sum()),
        "pcd_outside_head_envelope": int((labels == 0).sum()),
        **{f"pixels_{name}": int(value.sum()) for name, value in result.items()},
    }
    return result, counts

