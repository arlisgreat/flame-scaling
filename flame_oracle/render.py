from __future__ import annotations

import numpy as np
import torch


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    triangle = vertices[faces]
    face_normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-8)


def quaternions_from_z(normals: torch.Tensor) -> torch.Tensor:
    normals = torch.nn.functional.normalize(normals, dim=-1)
    quaternions = torch.stack(
        [1.0 + normals[:, 2], -normals[:, 1], normals[:, 0], torch.zeros_like(normals[:, 0])],
        dim=-1,
    )
    opposite = quaternions.norm(dim=-1) < 1e-6
    quaternions[opposite] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=normals.device)
    return torch.nn.functional.normalize(quaternions, dim=-1)

