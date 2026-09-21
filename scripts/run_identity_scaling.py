#!/usr/bin/env python3
"""Controlled cross-identity FLAME-mean scaling experiment.

This is deliberately a static multiview proxy.  Every condition receives the
same identity-specific, multiview-landmark-fitted FLAME scaffold, the same
semantic slots, observations, encoder, decoder, Gaussian count, and fixed
covariance.  The only intervention is whether predicted Gaussian means are
hard, sparsely/boundedly released, or broadly released from that scaffold.

The cache field ``initial_colors`` is never read: it was constructed from all
views and would leak evaluation targets into the observation encoder.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import pickle
import random
import time
from pathlib import Path
from typing import Any

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from skimage.metrics import structural_similarity

from flame_oracle.io import save_montage
from flame_oracle.metrics import alpha_metrics, masked_image_metrics
from flame_oracle.render import vertex_normals


DEFAULT_MANIFEST = Path("artifacts/audits/nersemble_identity_scaling.json")
DEFAULT_CACHE = Path("artifacts/cache/nersemble_scale_frames")
DEFAULT_MASKS = Path(
    "/data1/workspace/airulan/3dv/external/LAM/model_zoo/"
    "human_parametric_models/flame_vhap/FLAME_masks.pkl"
)


def camera_centers(viewmats: np.ndarray) -> np.ndarray:
    rotations = viewmats[:, :3, :3]
    translations = viewmats[:, :3, 3]
    return -np.einsum("vji,vj->vi", rotations, translations)


def greedy_farthest_camera_order(
    camera_ids: list[str], viewmats: np.ndarray, source_camera: str
) -> list[int]:
    """Source first, then a deterministic farthest angular prefix."""
    if source_camera not in camera_ids:
        raise ValueError(f"source camera {source_camera} is absent")
    centers = camera_centers(viewmats)
    directions = centers - centers.mean(axis=0, keepdims=True)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-8)
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    angular = np.arccos(cosine)
    selected = [camera_ids.index(source_camera)]
    remaining = set(range(len(camera_ids))) - set(selected)
    while remaining:
        chosen = max(
            remaining,
            key=lambda index: (
                float(np.min(angular[index, selected])),
                camera_ids[index],
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def bounded_vector(raw: torch.Tensor, radius_m: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    return radius_m * raw / torch.sqrt(1.0 + norm.square())


def unique_mesh_edges(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def surface_sampling_plan(
    faces: np.ndarray,
    original_vertex_count: int,
    density_multiplier: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create topology-consistent extra surface slots without using RGB targets."""
    if density_multiplier < 1:
        raise ValueError("density_multiplier must be at least one")
    extra_count = original_vertex_count * (density_multiplier - 1)
    if extra_count == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(
        len(faces), size=extra_count, replace=extra_count > len(faces)
    ).astype(np.int64)
    uv = rng.random((extra_count, 2))
    root = np.sqrt(uv[:, 0])
    barycentric = np.stack(
        [1.0 - root, root * (1.0 - uv[:, 1]), root * uv[:, 1]], axis=1
    ).astype(np.float32)
    return face_indices, barycentric


def augment_subject_density(
    subject: dict[str, Any], face_indices: np.ndarray, barycentric: np.ndarray
) -> None:
    """Append the same semantic face/barycentric slots to one identity scaffold."""
    original_count = len(subject["base"])
    subject["original_vertex_count"] = original_count
    if not len(face_indices):
        return
    device = subject["base"].device
    parents = torch.from_numpy(subject["faces"][face_indices]).long().to(device)
    weights = torch.from_numpy(barycentric).to(device=device, dtype=subject["base"].dtype)
    extra_base = (subject["base"][parents] * weights[..., None]).sum(dim=1)
    extra_normals = F.normalize(
        (subject["normals"][parents] * weights[..., None]).sum(dim=1), dim=1
    )
    subject["base"] = torch.cat([subject["base"], extra_base], dim=0)
    subject["normals"] = torch.cat([subject["normals"], extra_normals], dim=0)


def augmented_mesh_edges(
    faces: np.ndarray, original_vertex_count: int, face_indices: np.ndarray
) -> np.ndarray:
    edges = unique_mesh_edges(faces)
    if not len(face_indices):
        return edges
    parents = faces[face_indices]
    extra = original_vertex_count + np.arange(len(face_indices), dtype=np.int64)
    attachment = np.stack(
        [
            np.repeat(extra, 3),
            parents.reshape(-1),
        ],
        axis=1,
    )
    attachment.sort(axis=1)
    return np.unique(np.concatenate([edges, attachment], axis=0), axis=0)


def augment_region_indices(
    groups: dict[str, np.ndarray],
    faces: np.ndarray,
    original_vertex_count: int,
    face_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    if not len(face_indices):
        return groups
    parents = faces[face_indices]
    output: dict[str, np.ndarray] = {}
    for name, indices in groups.items():
        membership = np.zeros(original_vertex_count, dtype=bool)
        membership[np.asarray(indices, dtype=np.int64)] = True
        inherited = membership[parents].sum(axis=1) >= 2
        extra = original_vertex_count + np.flatnonzero(inherited)
        output[name] = np.concatenate([np.asarray(indices, dtype=np.int64), extra])
    return output


def released_means(
    base: torch.Tensor,
    raw_delta: torch.Tensor,
    raw_gate: torch.Tensor,
    method: str,
    adaptive_radius_mm: float,
    released_radius_mm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply only the pre-registered G_mu intervention."""
    gate = torch.sigmoid(raw_gate)
    if method == "hard":
        offset = 0.0 * raw_delta + 0.0 * raw_gate
    elif method == "adaptive":
        offset = gate * bounded_vector(raw_delta, adaptive_radius_mm / 1000.0)
    elif method == "released":
        offset = bounded_vector(raw_delta, released_radius_mm / 1000.0) + 0.0 * raw_gate
    else:
        raise ValueError(f"unknown method: {method}")
    return base + offset, offset, gate


def camera_features(
    viewmats: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    rotations = viewmats[:, :3, :3]
    translations = viewmats[:, :3, 3]
    centers = -torch.einsum("vji,vj->vi", rotations, translations)
    forward = rotations.transpose(1, 2)[:, :, 2]
    up = rotations.transpose(1, 2)[:, :, 1]
    calibration = torch.stack(
        [
            intrinsics[:, 0, 0] / width,
            intrinsics[:, 1, 1] / height,
            intrinsics[:, 0, 2] / width,
            intrinsics[:, 1, 2] / height,
        ],
        dim=1,
    )
    return torch.cat([centers, forward, up, calibration], dim=1)


def sample_vertex_observations(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    images: torch.Tensor,
    alpha: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Sample only observation images at the fixed semantic FLAME slots."""
    view_count, height, width = images.shape[:3]
    rotation = viewmats[:, :3, :3]
    translation = viewmats[:, :3, 3]
    camera = torch.einsum("vij,nj->vni", rotation, vertices) + translation[:, None]
    projected = torch.einsum("vij,vnj->vni", intrinsics, camera)
    pixels = projected[..., :2] / projected[..., 2:3].clamp_min(1e-6)
    grid = torch.stack(
        [
            2.0 * pixels[..., 0] / max(width - 1, 1) - 1.0,
            2.0 * pixels[..., 1] / max(height - 1, 1) - 1.0,
        ],
        dim=-1,
    )
    payload = torch.cat([images, alpha], dim=-1).permute(0, 3, 1, 2)
    sampled = F.grid_sample(
        payload,
        grid[:, None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, :, 0].permute(0, 2, 1)
    centers = -torch.einsum("vji,vj->vi", rotation, translation)
    to_camera = F.normalize(centers[:, None] - vertices[None], dim=-1)
    facing = torch.relu(torch.einsum("vni,ni->vn", to_camera, normals))
    valid = (
        (camera[..., 2] > 0)
        & (grid[..., 0] >= -1)
        & (grid[..., 0] <= 1)
        & (grid[..., 1] >= -1)
        & (grid[..., 1] <= 1)
    )
    weights = sampled[..., 3] * facing * valid.float()
    denominator = weights.sum(dim=0, keepdim=False)[:, None]
    mean = (weights[..., None] * sampled[..., :3]).sum(dim=0) / denominator.clamp_min(1e-6)
    mean = torch.where(denominator > 1e-6, mean, torch.full_like(mean, 0.5))
    variance = (
        weights[..., None] * (sampled[..., :3] - mean[None]).square()
    ).sum(dim=0) / denominator.clamp_min(1e-6)
    confidence = (denominator / max(float(view_count), 1.0)).clamp(0.0, 1.0)
    return torch.cat([mean, torch.sqrt(variance.clamp_min(1e-8)), confidence], dim=1)


class ObservationEncoder(torch.nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        channels = [4, 32, 64, 96, 128]
        blocks: list[torch.nn.Module] = []
        for left, right in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    torch.nn.Conv2d(left, right, 3, stride=2, padding=1),
                    torch.nn.GroupNorm(8, right),
                    torch.nn.SiLU(),
                ]
            )
        self.image_backbone = torch.nn.Sequential(*blocks)
        self.image_head = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 3, latent_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(latent_dim, latent_dim),
        )
        self.camera_head = torch.nn.Sequential(
            torch.nn.Linear(13, latent_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(latent_dim, latent_dim),
        )
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(latent_dim),
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.SiLU(),
        )

    def forward(
        self,
        images: torch.Tensor,
        alpha: torch.Tensor,
        camera: torch.Tensor,
    ) -> torch.Tensor:
        composite = images * alpha + (1.0 - alpha)
        inputs = torch.cat([composite, alpha], dim=-1).permute(0, 3, 1, 2)
        features = self.image_backbone(inputs)
        features = F.adaptive_avg_pool2d(features, (4, 3)).flatten(1)
        per_view = self.image_head(features) + self.camera_head(camera)
        return self.output(per_view.mean(dim=0, keepdim=True))[0]


class SharedGaussianReconstructor(torch.nn.Module):
    """One parameterization shared verbatim across every geometry condition."""

    def __init__(
        self,
        gaussian_count: int,
        latent_dim: int,
        slot_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(latent_dim)
        self.slot_embedding = torch.nn.Parameter(
            torch.randn(gaussian_count, slot_dim) / math.sqrt(slot_dim)
        )
        decoder_input = slot_dim + latent_dim + 3 + 3 + 7
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(decoder_input, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        self.color_head = torch.nn.Linear(hidden_dim, 3)
        self.opacity_head = torch.nn.Linear(hidden_dim, 1)
        self.delta_head = torch.nn.Linear(hidden_dim, 3)
        self.gate_head = torch.nn.Linear(hidden_dim, 1)
        torch.nn.init.zeros_(self.color_head.weight)
        torch.nn.init.zeros_(self.color_head.bias)
        torch.nn.init.constant_(self.opacity_head.bias, 2.0)
        torch.nn.init.normal_(self.delta_head.weight, std=1e-4)
        torch.nn.init.zeros_(self.delta_head.bias)
        torch.nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(
        self,
        base: torch.Tensor,
        normals: torch.Tensor,
        images: torch.Tensor,
        alpha: torch.Tensor,
        viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
    ) -> dict[str, torch.Tensor]:
        camera = camera_features(viewmats, intrinsics, width, height)
        latent = self.encoder(images, alpha, camera)
        observed = sample_vertex_observations(
            base, normals, images, alpha, viewmats, intrinsics
        )
        latent_slots = latent[None].expand(len(base), -1)
        inputs = torch.cat(
            [self.slot_embedding, latent_slots, base * 5.0, normals, observed], dim=1
        )
        trunk = self.trunk(inputs)
        observed_logit = torch.logit(observed[:, :3].clamp(1e-4, 1.0 - 1e-4))
        return {
            "colors": torch.sigmoid(observed_logit + self.color_head(trunk)),
            "opacities": torch.sigmoid(self.opacity_head(trunk)[:, 0]),
            "raw_delta": self.delta_head(trunk),
            "raw_gate": self.gate_head(trunk),
        }


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def load_subject(
    path: Path, common_camera_ids: list[str], device: torch.device
) -> dict[str, Any]:
    payload = np.load(path)
    # Intentionally do not access payload["initial_colors"].
    available = [str(value) for value in payload["camera_ids"].tolist()]
    missing = sorted(set(common_camera_ids) - set(available))
    if missing:
        raise RuntimeError(f"{path.stem} is missing common cameras {missing}")
    indices = np.asarray([available.index(camera) for camera in common_camera_ids])
    vertices = payload["vertices"].astype(np.float32)
    faces = payload["faces"].astype(np.int64)
    normals = vertex_normals(vertices, faces)
    return {
        "subject": str(payload["subject"].item()),
        "images_u8": _tensor(payload["images"][indices].astype(np.uint8), device),
        "alpha_u8": _tensor(payload["alpha"][indices].astype(np.uint8), device),
        "viewmats": _tensor(payload["viewmats"][indices].astype(np.float32), device),
        "Ks": _tensor(payload["Ks"][indices].astype(np.float32), device),
        "base": _tensor(vertices, device),
        "normals": _tensor(normals.astype(np.float32), device),
        "faces": faces,
        "landmarks": payload["target_landmarks"].astype(np.float32),
        "height": int(payload["images"].shape[1]),
        "width": int(payload["images"].shape[2]),
        "frame": int(payload["frame"][0]),
    }


def load_subjects(
    subjects: list[str], cache_root: Path, common_camera_ids: list[str], device: torch.device
) -> list[dict[str, Any]]:
    rows = []
    for subject in subjects:
        path = cache_root / f"{subject}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing scale cache: {path}")
        rows.append(load_subject(path, common_camera_ids, device))
    return rows


def float_views(subject: dict[str, Any], indices: list[int]) -> tuple[torch.Tensor, ...]:
    selected = torch.as_tensor(indices, device=subject["base"].device, dtype=torch.long)
    images = subject["images_u8"][selected].float() / 255.0
    alpha = subject["alpha_u8"][selected].float() / 255.0
    return images, alpha, subject["viewmats"][selected], subject["Ks"][selected]


def predict_attributes(
    model: SharedGaussianReconstructor,
    subject: dict[str, Any],
    observation_indices: list[int],
    method: str,
    adaptive_radius_mm: float,
    released_radius_mm: float,
) -> dict[str, torch.Tensor]:
    images, alpha, viewmats, intrinsics = float_views(subject, observation_indices)
    attributes = model(
        subject["base"],
        subject["normals"],
        images,
        alpha,
        viewmats,
        intrinsics,
        subject["width"],
        subject["height"],
    )
    means, offsets, gate = released_means(
        subject["base"],
        attributes["raw_delta"],
        attributes["raw_gate"],
        method,
        adaptive_radius_mm,
        released_radius_mm,
    )
    return {**attributes, "means": means, "offsets": offsets, "gate": gate}


def render_attributes(
    attributes: dict[str, torch.Tensor],
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    fixed_scale_mm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(attributes["means"])
    quaternions = torch.zeros((count, 4), device=attributes["means"].device)
    quaternions[:, 0] = 1.0
    scales = torch.full(
        (count, 3), fixed_scale_mm / 1000.0, device=attributes["means"].device
    )
    rendered, alpha, _ = rasterization(
        attributes["means"],
        quaternions,
        scales,
        attributes["opacities"],
        attributes["colors"],
        viewmats,
        intrinsics,
        width,
        height,
        packed=True,
    )
    return rendered + (1.0 - alpha), alpha


def project_points(
    points: np.ndarray, viewmat: np.ndarray, intrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    camera = (viewmat @ homogeneous.T).T[:, :3]
    projected = (intrinsic @ camera.T).T
    pixels = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
    return pixels, camera[:, 2]


def mesh_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    viewmat: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    pixels, depth = project_points(vertices, viewmat, intrinsic)
    triangles = pixels[faces]
    valid = np.all(depth[faces] > 0, axis=1) & np.isfinite(triangles).all(axis=(1, 2))
    triangles = np.rint(triangles[valid]).astype(np.int32)
    if len(triangles):
        low = triangles.min(axis=1)
        high = triangles.max(axis=1)
        intersects = (
            (high[:, 0] >= 0)
            & (high[:, 1] >= 0)
            & (low[:, 0] < width)
            & (low[:, 1] < height)
        )
        triangles = triangles[intersects]
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(triangles):
        cv2.fillPoly(mask, list(triangles), 1)
    return mask.astype(bool)


def polygon_mask(
    points: np.ndarray, height: int, width: int, dilation: int = 1
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if np.isfinite(points).all():
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    if dilation > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=dilation)
    return mask.astype(bool)


def evaluation_masks(
    subject: dict[str, Any], camera_index: int, target_alpha: np.ndarray
) -> dict[str, np.ndarray]:
    viewmat = subject["viewmats"][camera_index].detach().cpu().numpy()
    intrinsic = subject["Ks"][camera_index].detach().cpu().numpy()
    height, width = target_alpha.shape[:2]
    vertices = subject["base"].detach().cpu().numpy()
    support = mesh_silhouette(
        vertices, subject["faces"], viewmat, intrinsic, height, width
    )
    support = cv2.dilate(
        support.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool)
    landmark_pixels, landmark_depth = project_points(
        subject["landmarks"], viewmat, intrinsic
    )
    foreground = target_alpha[..., 0] > 0.05
    if np.all(landmark_depth[48:60] > 0):
        oral = polygon_mask(landmark_pixels[48:60], height, width, dilation=1)
    else:
        oral = np.zeros((height, width), dtype=bool)
    eyes = np.zeros((height, width), dtype=bool)
    if np.all(landmark_depth[36:48] > 0):
        eyes = polygon_mask(landmark_pixels[36:42], height, width, dilation=1)
        eyes |= polygon_mask(landmark_pixels[42:48], height, width, dilation=1)
    return {
        "full_frame": np.ones((height, width), dtype=bool),
        "foreground": foreground,
        "flame_supported": foreground & support,
        "flame_unsupported": foreground & ~support,
        "oral": foreground & oral,
        "eyes": foreground & eyes,
        "background": ~foreground,
    }


def quick_validation(
    model: SharedGaussianReconstructor,
    subjects: list[dict[str, Any]],
    camera_order: list[int],
    observation_count: int,
    method: str,
    adaptive_radius_mm: float,
    released_radius_mm: float,
    fixed_scale_mm: float,
    training_region: str,
) -> float:
    if not subjects:
        return float("nan")
    observation = camera_order[:observation_count]
    targets = camera_order[observation_count:]
    values = []
    model.eval()
    with torch.no_grad():
        for subject in subjects:
            attributes = predict_attributes(
                model,
                subject,
                observation,
                method,
                adaptive_radius_mm,
                released_radius_mm,
            )
            target_tensor = torch.as_tensor(targets, device=subject["base"].device)
            rendered, _ = render_attributes(
                attributes,
                subject["viewmats"][target_tensor],
                subject["Ks"][target_tensor],
                subject["width"],
                subject["height"],
                fixed_scale_mm,
            )
            target_rgb = subject["images_u8"][target_tensor].float() / 255.0
            target_alpha = subject["alpha_u8"][target_tensor].float() / 255.0
            target_rgb = target_rgb * target_alpha + (1.0 - target_alpha)
            validation_weight = target_alpha
            if training_region == "head_roi":
                validation_weight = validation_weight * subject["training_head_roi_u8"][
                    target_tensor
                ].float()
            denominator = (validation_weight.sum() * 3.0).clamp_min(1.0)
            values.append(
                float(
                    (torch.abs(rendered - target_rgb) * validation_weight).sum()
                    / denominator
                )
            )
    model.train()
    return float(np.mean(values))


def mask_region_indices(mask_path: Path) -> dict[str, np.ndarray]:
    if not mask_path.exists():
        return {}
    with mask_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    groups = {
        "face": ["face", "forehead", "nose"],
        "oral_slots": ["lips"],
        "eye_slots": ["left_eye_region", "right_eye_region"],
        "scalp": ["scalp"],
        "ears": ["left_ear", "right_ear"],
        "neck": ["neck"],
    }
    return {
        name: np.unique(np.concatenate([np.asarray(payload[key], dtype=np.int64) for key in keys]))
        for name, keys in groups.items()
    }


def projected_head_envelope(
    subject: dict[str, Any],
    camera_index: int,
    head_indices: np.ndarray,
    dilation_px: int,
) -> np.ndarray:
    viewmat = subject["viewmats"][camera_index].detach().cpu().numpy()
    intrinsic = subject["Ks"][camera_index].detach().cpu().numpy()
    vertices = subject["base"][head_indices].detach().cpu().numpy()
    pixels, depth = project_points(vertices, viewmat, intrinsic)
    envelope = np.zeros((subject["height"], subject["width"]), dtype=np.uint8)
    visible = (depth > 0) & np.isfinite(pixels).all(axis=1)
    if visible.sum() >= 3:
        hull = cv2.convexHull(np.rint(pixels[visible]).astype(np.int32))
        cv2.fillConvexPoly(envelope, hull, 1)
    if dilation_px > 0:
        size = 2 * dilation_px + 1
        envelope = cv2.dilate(
            envelope, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        )
    return envelope.astype(bool)


def attach_head_training_masks(
    subjects: list[dict[str, Any]],
    groups: dict[str, np.ndarray],
    dilation_px: int,
    background_ring_px: int,
) -> None:
    head_indices = np.unique(
        np.concatenate(
            [groups[key] for key in ["face", "oral_slots", "eye_slots", "scalp", "ears"]]
        )
    )
    for subject in subjects:
        envelopes = []
        rings = []
        for camera_index in range(len(subject["viewmats"])):
            envelope = projected_head_envelope(
                subject, camera_index, head_indices, dilation_px
            )
            if background_ring_px > 0:
                size = 2 * background_ring_px + 1
                outer = cv2.dilate(
                    envelope.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                ).astype(bool)
                ring = outer & ~envelope
            else:
                ring = ~envelope
            envelopes.append(envelope[..., None].astype(np.uint8))
            rings.append(ring[..., None].astype(np.uint8))
        device = subject["base"].device
        subject["training_head_roi_u8"] = _tensor(np.stack(envelopes), device)
        subject["training_head_ring_u8"] = _tensor(np.stack(rings), device)


def geometry_summary(
    attributes: dict[str, torch.Tensor], region_indices: dict[str, np.ndarray]
) -> dict[str, Any]:
    offsets = torch.linalg.vector_norm(attributes["offsets"], dim=1).detach().cpu().numpy() * 1000.0
    gates = attributes["gate"].detach().cpu().numpy()[:, 0]

    def summarize(indices: np.ndarray | None = None) -> dict[str, float]:
        selected_offsets = offsets if indices is None else offsets[indices]
        selected_gates = gates if indices is None else gates[indices]
        return {
            "count": int(len(selected_offsets)),
            "offset_mean_mm": float(selected_offsets.mean()),
            "offset_median_mm": float(np.median(selected_offsets)),
            "offset_p90_mm": float(np.quantile(selected_offsets, 0.9)),
            "offset_max_mm": float(selected_offsets.max()),
            "fraction_offset_gt_5mm": float(np.mean(selected_offsets > 5.0)),
            "gate_mean": float(selected_gates.mean()),
            "fraction_gate_gt_half": float(np.mean(selected_gates > 0.5)),
        }

    return {
        "all": summarize(),
        "regions": {name: summarize(indices) for name, indices in region_indices.items()},
    }


def evaluate_test(
    model: SharedGaussianReconstructor,
    subjects: list[dict[str, Any]],
    camera_ids: list[str],
    camera_order: list[int],
    observation_counts: list[int],
    method: str,
    adaptive_radius_mm: float,
    released_radius_mm: float,
    fixed_scale_mm: float,
    region_indices: dict[str, np.ndarray],
    output: Path,
    audit_subjects: set[str],
    perceptual_model: torch.nn.Module,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for subject in subjects:
            for observation_count in observation_counts:
                observation = camera_order[:observation_count]
                targets = camera_order[observation_count:]
                attributes = predict_attributes(
                    model,
                    subject,
                    observation,
                    method,
                    adaptive_radius_mm,
                    released_radius_mm,
                )
                geometry_rows.append(
                    {
                        "subject": subject["subject"],
                        "n_obs": observation_count,
                        **geometry_summary(attributes, region_indices),
                    }
                )
                target_tensor = torch.as_tensor(targets, device=subject["base"].device)
                rendered, predicted_alpha = render_attributes(
                    attributes,
                    subject["viewmats"][target_tensor],
                    subject["Ks"][target_tensor],
                    subject["width"],
                    subject["height"],
                    fixed_scale_mm,
                )
                rendered_np = rendered.cpu().numpy()
                predicted_alpha_np = predicted_alpha.cpu().numpy()
                images = subject["images_u8"][target_tensor].float().cpu().numpy() / 255.0
                target_alpha = subject["alpha_u8"][target_tensor].float().cpu().numpy() / 255.0
                target_composite = images * target_alpha + (1.0 - target_alpha)
                target_composite_t = torch.from_numpy(target_composite).to(rendered.device)
                perceptual_map = perceptual_model(
                    rendered.permute(0, 3, 1, 2) * 2.0 - 1.0,
                    target_composite_t.permute(0, 3, 1, 2) * 2.0 - 1.0,
                ).detach().cpu().numpy()[:, 0]
                if perceptual_map.shape[1:] != (subject["height"], subject["width"]):
                    perceptual_map = np.stack(
                        [
                            cv2.resize(
                                value,
                                (subject["width"], subject["height"]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            for value in perceptual_map
                        ]
                    )
                montage: list[np.ndarray] = []
                for local_index, camera_index in enumerate(targets):
                    masks = evaluation_masks(subject, camera_index, target_alpha[local_index])
                    for region, mask in masks.items():
                        metrics = masked_image_metrics(
                            rendered_np[local_index], target_composite[local_index], mask
                        )
                        alpha_values = alpha_metrics(
                            predicted_alpha_np[local_index] * mask[..., None],
                            target_alpha[local_index] * mask[..., None],
                        )
                        row = {
                            "subject": subject["subject"],
                            "n_obs": observation_count,
                            "camera_id": camera_ids[camera_index],
                            "region": region,
                            "pixel_count": int(mask.sum()),
                            "lpips_alex_spatial": float(
                                (perceptual_map[local_index] * mask).sum()
                                / max(float(mask.sum()), 1.0)
                            ),
                            **metrics,
                            **alpha_values,
                        }
                        if region == "full_frame":
                            row["ssim"] = float(
                                structural_similarity(
                                    target_composite[local_index],
                                    rendered_np[local_index],
                                    channel_axis=2,
                                    data_range=1.0,
                                )
                            )
                        rows.append(row)
                    if subject["subject"] in audit_subjects and local_index < 6:
                        montage.extend([target_composite[local_index], rendered_np[local_index]])
                if montage:
                    save_montage(
                        output
                        / "audits"
                        / f"{subject['subject']}_nobs{observation_count}_target_prediction.png",
                        montage,
                        columns=4,
                    )
    return rows, geometry_rows


def count_parameters(
    model: SharedGaussianReconstructor, method: str
) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    delta = sum(parameter.numel() for parameter in model.delta_head.parameters())
    gate = sum(parameter.numel() for parameter in model.gate_head.parameters())
    inactive = delta + gate if method == "hard" else (gate if method == "released" else 0)
    return {
        "total_trainable": int(total),
        "active_under_intervention": int(total - inactive),
        "dormant_g_mu_head": int(delta if method == "hard" else 0),
        "dormant_gate_head": int(gate if method in {"hard", "released"} else 0),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row if not isinstance(row[key], dict)})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--n-id", type=int, required=True)
    parser.add_argument("--method", choices=["hard", "adaptive", "released"], required=True)
    parser.add_argument(
        "--adaptive-radius-mm",
        type=float,
        default=150.0,
        help="Maximum APR displacement before the learned sparse gate; matched to released by default.",
    )
    parser.add_argument("--released-radius-mm", type=float, default=150.0)
    parser.add_argument(
        "--fixed-scale-mm",
        type=float,
        default=5.0,
        help="Fixed isotropic sigma; 5 mm is the pre-matrix coverage-calibrated primary value.",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--offset-magnitude-weight", type=float, default=0.01)
    parser.add_argument("--offset-laplacian-weight", type=float, default=2.0)
    parser.add_argument("--adaptive-gate-weight", type=float, default=0.003)
    parser.add_argument("--adaptive-gate-laplacian-weight", type=float, default=0.1)
    parser.add_argument("--observation-counts", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--validation-observation-count", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--slot-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--training-region", choices=["full_frame", "head_roi"], default="full_frame"
    )
    parser.add_argument("--head-roi-dilation-px", type=int, default=6)
    parser.add_argument("--head-background-ring-px", type=int, default=6)
    parser.add_argument("--density-multiplier", type=int, default=1)
    parser.add_argument("--surface-sampling-seed", type=int, default=20260716)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--audit-subjects", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.fixed_scale_mm <= 0:
        parser.error("--fixed-scale-mm must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.density_multiplier < 1:
        parser.error("--density-multiplier must be at least one")
    if args.head_roi_dilation_px < 0 or args.head_background_ring_px < 0:
        parser.error("head ROI dilation and ring widths must be non-negative")
    if any(value <= 0 for value in args.observation_counts):
        parser.error("observation counts must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text())
    train_key = str(args.n_id)
    if train_key not in manifest["nested_train_sets"]:
        parser.error(f"n-id {args.n_id} is absent from the manifest")
    train_ids = manifest["nested_train_sets"][train_key]
    validation_ids = manifest.get("validation_subjects", [])
    if args.validation_limit is not None:
        validation_ids = validation_ids[: args.validation_limit]
    test_ids = manifest["usable_svfr_test"]
    if args.test_limit is not None:
        test_ids = test_ids[: args.test_limit]
    overlap = (set(train_ids) & set(validation_ids)) | (set(train_ids) & set(test_ids))
    if overlap:
        raise RuntimeError(f"identity leakage across splits: {sorted(overlap)}")
    common_camera_ids = manifest["common_camera_ids"]
    if max(args.observation_counts + [args.validation_observation_count]) >= len(common_camera_ids):
        parser.error("every observation count must leave at least one target camera")

    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train_subjects = load_subjects(train_ids, args.cache_root, common_camera_ids, device)
    validation_subjects = load_subjects(
        validation_ids, args.cache_root, common_camera_ids, device
    )
    test_subjects = load_subjects(test_ids, args.cache_root, common_camera_ids, device)
    reference = train_subjects[0]
    original_vertex_count = len(reference["base"])
    surface_face_indices, surface_barycentric = surface_sampling_plan(
        reference["faces"],
        original_vertex_count,
        args.density_multiplier,
        args.surface_sampling_seed,
    )
    for subject in train_subjects + validation_subjects + test_subjects:
        augment_subject_density(subject, surface_face_indices, surface_barycentric)
    base_region_indices = mask_region_indices(args.mask_path)
    if args.training_region == "head_roi":
        attach_head_training_masks(
            train_subjects + validation_subjects,
            base_region_indices,
            args.head_roi_dilation_px,
            args.head_background_ring_px,
        )
    camera_order = greedy_farthest_camera_order(
        common_camera_ids,
        reference["viewmats"].detach().cpu().numpy(),
        manifest["source_camera"],
    )
    gaussian_count = len(reference["base"])
    if any(len(subject["base"]) != gaussian_count for subject in train_subjects + validation_subjects + test_subjects):
        raise RuntimeError("all identity scaffolds must have the same slot count")
    mesh_edges = torch.from_numpy(
        augmented_mesh_edges(
            reference["faces"], original_vertex_count, surface_face_indices
        )
    ).long().to(device)

    model = SharedGaussianReconstructor(
        gaussian_count,
        args.latent_dim,
        args.slot_dim,
        args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05
    )
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    observation_draw_counts = {str(value): 0 for value in args.observation_counts}
    subject_draw_counts = {subject: 0 for subject in train_ids}
    target_camera_draw_counts = {camera: 0 for camera in common_camera_ids}
    model.train()

    for step in range(1, args.steps + 1):
        subject = train_subjects[int(rng.integers(len(train_subjects)))]
        observation_count = int(rng.choice(args.observation_counts))
        observation = camera_order[:observation_count]
        target_candidates = camera_order[observation_count:]
        target_index = int(rng.choice(target_candidates))
        observation_draw_counts[str(observation_count)] += 1
        subject_draw_counts[subject["subject"]] += 1
        target_camera_draw_counts[common_camera_ids[target_index]] += 1
        optimizer.zero_grad(set_to_none=True)
        attributes = predict_attributes(
            model,
            subject,
            observation,
            args.method,
            args.adaptive_radius_mm,
            args.released_radius_mm,
        )
        target_tensor = torch.as_tensor([target_index], device=device)
        rendered, predicted_alpha = render_attributes(
            attributes,
            subject["viewmats"][target_tensor],
            subject["Ks"][target_tensor],
            subject["width"],
            subject["height"],
            args.fixed_scale_mm,
        )
        target_rgb = subject["images_u8"][target_tensor].float() / 255.0
        target_alpha = subject["alpha_u8"][target_tensor].float() / 255.0
        target_rgb = target_rgb * target_alpha + (1.0 - target_alpha)
        rgb_weight = 0.25 + 0.75 * target_alpha
        if args.training_region == "head_roi":
            roi = subject["training_head_roi_u8"][target_tensor].float()
            rgb_weight = rgb_weight * roi
        rgb_loss = (
            torch.abs(rendered - target_rgb) * rgb_weight
        ).sum() / (rgb_weight.sum() * 3.0).clamp_min(1.0)
        if args.training_region == "head_roi":
            alpha_loss = (
                torch.abs(predicted_alpha - target_alpha) * roi
            ).sum() / roi.sum().clamp_min(1.0)
            background = (
                (target_alpha < 0.05)
                & subject["training_head_ring_u8"][target_tensor].bool()
            )
        else:
            alpha_loss = torch.abs(predicted_alpha - target_alpha).mean()
            background = target_alpha < 0.05
        leak_loss = (predicted_alpha * background).sum() / background.sum().clamp_min(1.0)
        radius_m = (
            args.adaptive_radius_mm if args.method == "adaptive" else args.released_radius_mm
        ) / 1000.0
        normalized_offset = attributes["offsets"] / max(radius_m, 1e-8)
        offset_magnitude_penalty = normalized_offset.square().sum(dim=1).mean()
        edge_difference = normalized_offset[mesh_edges[:, 0]] - normalized_offset[mesh_edges[:, 1]]
        offset_laplacian_penalty = edge_difference.square().sum(dim=1).mean()
        gate_laplacian_penalty = (
            attributes["gate"][mesh_edges[:, 0]] - attributes["gate"][mesh_edges[:, 1]]
        ).square().mean()
        if args.method == "adaptive":
            release_regularizer = (
                args.offset_magnitude_weight * offset_magnitude_penalty
                + args.offset_laplacian_weight * offset_laplacian_penalty
                + args.adaptive_gate_weight * attributes["gate"].mean()
                + args.adaptive_gate_laplacian_weight * gate_laplacian_penalty
            )
        elif args.method == "released":
            release_regularizer = (
                args.offset_magnitude_weight * offset_magnitude_penalty
                + args.offset_laplacian_weight * offset_laplacian_penalty
            )
        else:
            release_regularizer = 0.0 * normalized_offset.mean()
        loss = rgb_loss + 0.2 * alpha_loss + 0.05 * leak_loss + release_regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        should_log = step == 1 or step % args.log_every == 0 or step == args.steps
        should_evaluate = (
            step == args.steps
            or (args.eval_every > 0 and step % args.eval_every == 0)
        )
        record: dict[str, Any] | None = None
        if should_log or should_evaluate:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "rgb_l1": float(rgb_loss.detach()),
                "alpha_l1": float(alpha_loss.detach()),
                "background_leak": float(leak_loss.detach()),
                "release_regularizer": float(release_regularizer.detach()),
                "offset_magnitude_penalty": float(offset_magnitude_penalty.detach()),
                "offset_laplacian_penalty": float(offset_laplacian_penalty.detach()),
                "gate_laplacian_penalty": float(gate_laplacian_penalty.detach()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "sample_n_obs": observation_count,
                "sample_subject": subject["subject"],
                "mean_offset_mm": float(
                    torch.linalg.vector_norm(attributes["offsets"], dim=1).mean().detach() * 1000.0
                ),
                "mean_gate": float(attributes["gate"].mean().detach()),
                "elapsed_seconds": time.time() - start,
            }
        if should_evaluate:
            validation = quick_validation(
                model,
                validation_subjects,
                camera_order,
                args.validation_observation_count,
                args.method,
                args.adaptive_radius_mm,
                args.released_radius_mm,
                args.fixed_scale_mm,
                args.training_region,
            )
            assert record is not None
            record["validation_foreground_mae"] = validation
            if validation < best_validation or best_state is None:
                best_validation = validation
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if record is not None:
            history.append(record)
            print(json.dumps(record), flush=True)

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_step = args.steps
    model.load_state_dict(best_state)
    perceptual_model = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()
    for parameter in perceptual_model.parameters():
        parameter.requires_grad_(False)
    region_indices = augment_region_indices(
        base_region_indices,
        reference["faces"],
        original_vertex_count,
        surface_face_indices,
    )
    test_rows, geometry_rows = evaluate_test(
        model,
        test_subjects,
        common_camera_ids,
        camera_order,
        sorted(set(args.observation_counts)),
        args.method,
        args.adaptive_radius_mm,
        args.released_radius_mm,
        args.fixed_scale_mm,
        region_indices,
        args.output,
        set(args.audit_subjects),
        perceptual_model,
    )
    write_rows(args.output / "test_rows.csv", test_rows)
    torch.save(
        {
            "state_dict": best_state,
            "n_id": args.n_id,
            "method": args.method,
            "best_step": best_step,
            "camera_order": camera_order,
            "common_camera_ids": common_camera_ids,
        },
        args.output / "checkpoint.pt",
    )
    report = {
        "kind": "nersemble_static_cross_identity_flame_mean_scaling",
        "method": args.method,
        "n_id": args.n_id,
        "seed": args.seed,
        "train_subjects": train_ids,
        "validation_subjects": validation_ids,
        "test_subjects": test_ids,
        "official_test_protocol": "NeRSemble SVFR identities held out from every train level",
        "camera_ids": common_camera_ids,
        "camera_order": [common_camera_ids[index] for index in camera_order],
        "observation_counts": sorted(set(args.observation_counts)),
        "observation_contract": (
            "Nested camera prefixes beginning at source camera; every evaluated target camera is absent "
            "from that prefix. Training targets are also sampled only from the complement."
        ),
        "geometry_intervention": {
            "adaptive_radius_mm": args.adaptive_radius_mm,
            "released_radius_mm": args.released_radius_mm,
            "gaussian_count": gaussian_count,
            "original_vertex_count": original_vertex_count,
            "density_multiplier": args.density_multiplier,
            "surface_sampling_seed": args.surface_sampling_seed,
            "surface_sample_count": int(len(surface_face_indices)),
            "surface_sampling_sha256": hashlib.sha256(
                surface_face_indices.tobytes() + surface_barycentric.tobytes()
            ).hexdigest(),
            "fixed_isotropic_scale_mm": args.fixed_scale_mm,
            "fixed_semantic_correspondence": True,
            "fixed_multiview_oracle_flame_initialization": True,
        },
        "architecture": {
            "latent_dim": args.latent_dim,
            "slot_dim": args.slot_dim,
            "hidden_dim": args.hidden_dim,
            "parameters": count_parameters(model, args.method),
        },
        "optimization": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "schedule": "cosine_to_0.05x",
            "offset_magnitude_weight": args.offset_magnitude_weight,
            "offset_laplacian_weight": args.offset_laplacian_weight,
            "adaptive_gate_weight": args.adaptive_gate_weight,
            "adaptive_gate_laplacian_weight": args.adaptive_gate_laplacian_weight,
            "mesh_edge_count": int(len(mesh_edges)),
            "training_region": args.training_region,
            "head_roi_dilation_px": args.head_roi_dilation_px,
            "head_background_ring_px": args.head_background_ring_px,
            "validation_selection_region": (
                "head_foreground" if args.training_region == "head_roi" else "foreground"
            ),
            "eval_every": args.eval_every,
            "best_validation_step": best_step,
            "best_validation_foreground_mae": best_validation,
            "observation_draw_counts": observation_draw_counts,
            "subject_draw_counts": subject_draw_counts,
            "subject_draw_count_summary": {
                "minimum": int(min(subject_draw_counts.values())),
                "median": float(np.median(list(subject_draw_counts.values()))),
                "maximum": int(max(subject_draw_counts.values())),
            },
            "target_camera_draw_counts": target_camera_draw_counts,
        },
        "history": history,
        "test_rows": test_rows,
        "geometry_rows": geometry_rows,
        "leakage_controls": {
            "cache_initial_colors_read": False,
            "target_images_used_by_encoder": False,
            "test_identities_in_training": False,
            "validation_identities_in_training": False,
        },
        "perceptual_metric": {
            "name": "LPIPS",
            "backbone": "AlexNet",
            "package_version": "0.1.4",
            "region_reduction": "mean of the LPIPS spatial map over the pre-registered pixel mask",
            "used_for_checkpoint_selection": False,
        },
        "scope_warning": (
            "Static cross-identity proxy, not a full dynamic avatar system. The identity-specific FLAME "
            "base is fitted from multiview triangulated landmarks, including target cameras, so it is an "
            "oracle geometric scaffold deliberately favorable to the hard prior. RGB targets remain "
            "strictly held out. The released condition relaxes G_mu only and retains FLAME initialization "
            "and semantic correspondence C; it is not a fully FLAME-free model."
        ),
        "runtime_seconds": time.time() - start,
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024.0**3),
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "method": args.method,
                "n_id": args.n_id,
                "best_step": best_step,
                "best_validation_foreground_mae": best_validation,
                "test_row_count": len(test_rows),
                "runtime_seconds": report["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
