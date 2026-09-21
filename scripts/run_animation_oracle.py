#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

from flame_oracle.flame import (
    apply_dynamic_flame,
    apply_tracking_world_transform,
    load_flame_dynamic_assets,
    load_flame_linear_assets,
)
from flame_oracle.io import read_video_frame, save_montage, save_rgb
from flame_oracle.metrics import alpha_metrics, masked_image_metrics
from scripts.build_visual_hull import project


DEFAULT_ASSET_ROOT = Path(
    "/data1/workspace/airulan/3dv/external/LAM/model_zoo/human_parametric_models/flame_vhap"
)


def logit(value: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    value = np.clip(value, epsilon, 1.0 - epsilon)
    return np.log(value / (1.0 - value))


def nested_uniform_subset(indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or count >= len(indices):
        return indices.copy()
    positions = np.linspace(0, len(indices) - 1, count)
    selected = np.unique(np.round(positions).astype(np.int64))
    if len(selected) != count:
        missing = [index for index in range(len(indices)) if index not in set(selected)]
        selected = np.sort(np.r_[selected, missing[: count - len(selected)]])
    return indices[selected]


def tracking_codes(tracking: np.lib.npyio.NpzFile, latent_dimensions: int) -> tuple[np.ndarray, dict[str, object]]:
    features = np.concatenate(
        [tracking["expression"], tracking["neck"], tracking["jaw"], tracking["eyes"]], axis=1
    ).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    standard_deviation = features.std(axis=0, keepdims=True)
    standardized = (features - mean) / np.maximum(standard_deviation, 1e-4)
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    retained = min(latent_dimensions, right.shape[0])
    scores = standardized @ right[:retained].T
    score_scale = np.maximum(scores.std(axis=0, keepdims=True), 1e-4)
    scores = scores / score_scale
    if retained < latent_dimensions:
        scores = np.pad(scores, ((0, 0), (0, latent_dimensions - retained)))
    codes = np.concatenate([np.ones((len(scores), 1), np.float32), scores.astype(np.float32)], axis=1)
    report = {
        "raw_dimensions": int(features.shape[1]),
        "latent_dimensions": latent_dimensions,
        "retained_rank": retained,
        "explained_variance_fraction": float(
            np.square(singular[:retained]).sum() / max(float(np.square(singular).sum()), 1e-12)
        ),
        "uses_image_targets": False,
        "fit_scope": "all tracking codes in the sequence; no RGB or alpha",
    }
    return codes, report


def bounded_offsets(raw: torch.Tensor, radius_mm: float) -> torch.Tensor:
    norms = raw.norm(dim=-1, keepdim=True)
    return (radius_mm / 1000.0) * raw / torch.sqrt(1.0 + norms.square())


def local_means_for_method(
    method: str,
    hard_local: torch.Tensor,
    neutral_local: torch.Tensor,
    codes: torch.Tensor,
    deformation_basis: torch.Tensor,
    release_logits: torch.Tensor,
    radius_mm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.einsum("tk,kvj->tvj", codes, deformation_basis)
    gates = torch.sigmoid(release_logits)[None, :, None]
    if method == "hard":
        means = hard_local + 0.0 * raw + 0.0 * gates
    elif method == "bounded":
        means = hard_local + bounded_offsets(raw, radius_mm)
    elif method == "adaptive":
        means = hard_local + gates * bounded_offsets(raw, radius_mm)
    elif method == "free":
        means = neutral_local[None] + raw
    else:
        raise ValueError(f"unknown animation method: {method}")
    return means, raw, gates


def convex_region_mask(
    vertices: np.ndarray,
    vertex_ids: np.ndarray,
    viewmat: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    dilation: int = 0,
) -> np.ndarray:
    u, v, z = project(vertices[vertex_ids], viewmat, intrinsic)
    valid = (z > 0) & np.isfinite(u) & np.isfinite(v)
    points = np.stack([u[valid], v[valid]], axis=1)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(points).astype(np.int32)), 255)
    if dilation > 0:
        kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask.astype(np.float32) / 255.0


def build_semantic_masks(
    hard_world: np.ndarray,
    masks: dict[str, np.ndarray],
    viewmat: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    head_ids = np.unique(
        np.concatenate([masks[name] for name in ("face", "scalp", "left_ear", "right_ear", "neck")])
    )
    dilation = max(2, round(height / 64))
    outputs = {"head": [], "face": [], "oral": [], "eyes": []}
    for vertices in hard_world:
        outputs["head"].append(
            convex_region_mask(vertices, head_ids, viewmat, intrinsic, height, width, dilation)
        )
        outputs["face"].append(
            convex_region_mask(vertices, masks["face"], viewmat, intrinsic, height, width, dilation // 2)
        )
        outputs["oral"].append(
            convex_region_mask(vertices, masks["lips"], viewmat, intrinsic, height, width, dilation * 2)
        )
        left = convex_region_mask(
            vertices, masks["left_eye_region"], viewmat, intrinsic, height, width, dilation
        )
        right = convex_region_mask(
            vertices, masks["right_eye_region"], viewmat, intrinsic, height, width, dilation
        )
        outputs["eyes"].append(np.maximum(left, right))
    return {name: np.stack(values)[..., None] for name, values in outputs.items()}


def sample_initial_colors(
    images: np.ndarray,
    alpha: np.ndarray,
    vertices: np.ndarray,
    viewmat: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    height, width = images.shape[1:3]
    samples = []
    for image, target_alpha, points in zip(images, alpha, vertices):
        u, v, z = project(points, viewmat, intrinsic)
        ui = np.clip(np.round(u).astype(np.int64), 0, width - 1)
        vi = np.clip(np.round(v).astype(np.int64), 0, height - 1)
        valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid &= target_alpha[vi, ui, 0] > 0.5
        values = np.full((len(points), 3), np.nan, dtype=np.float32)
        values[valid] = image[vi[valid], ui[valid]]
        samples.append(values)
    with np.errstate(all="ignore"):
        colors = np.nanmedian(np.stack(samples), axis=0)
    return np.nan_to_num(colors, nan=0.5).astype(np.float32)


def render_frames(
    world_means: torch.Tensor,
    quaternions: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    intrinsic: torch.Tensor,
    width: int,
    height: int,
    attribute_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered, alphas = [], []
    for frame_position, means in enumerate(world_means):
        indices = attribute_indices[frame_position] if attribute_indices is not None else None
        rgb, alpha, _ = rasterization(
            means,
            quaternions if indices is None else quaternions[indices],
            scales if indices is None else scales[indices],
            opacities if indices is None else opacities[indices],
            colors if indices is None else colors[indices],
            viewmat[None],
            intrinsic[None],
            width,
            height,
            packed=True,
        )
        rendered.append(rgb[0] + (1.0 - alpha[0]))
        alphas.append(alpha[0])
    return torch.stack(rendered), torch.stack(alphas)


def build_correspondence_permutations(
    neutral_vertices: np.ndarray,
    frame_count: int,
    fraction: float,
    mode: str,
    local_bin_mm: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Create deterministic per-frame attribute-to-geometry assignments.

    Geometry centers are untouched.  Only Gaussian attribute identity is
    permuted, which makes this a controlled intervention on correspondence C.
    """
    vertex_count = len(neutral_vertices)
    identity = np.arange(vertex_count, dtype=np.int64)
    if fraction <= 0 or mode == "consistent":
        return np.tile(identity[None], (frame_count, 1)), 0.0
    if not 0 < fraction <= 1:
        raise ValueError("correspondence shuffle fraction must be in [0,1]")
    if mode == "global":
        groups = [identity]
    elif mode == "local":
        bins = np.floor(neutral_vertices / (local_bin_mm / 1000.0)).astype(np.int64)
        _, inverse = np.unique(bins, axis=0, return_inverse=True)
        groups = [np.flatnonzero(inverse == group) for group in range(inverse.max() + 1)]
    else:
        raise ValueError(f"unknown correspondence mode: {mode}")

    permutations = np.tile(identity[None], (frame_count, 1))
    for frame in range(frame_count):
        rng = np.random.default_rng(seed + 1000003 * frame)
        for group in groups:
            selected_count = min(len(group), max(0, round(fraction * len(group))))
            if selected_count < 2:
                continue
            selected = rng.choice(group, selected_count, replace=False)
            shift = int(rng.integers(1, selected_count))
            permutations[frame, selected] = np.roll(selected, shift)
    changed_fraction = float(np.mean(permutations != identity[None]))
    return permutations, changed_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--method", choices=("hard", "bounded", "adaptive", "free"), required=True)
    parser.add_argument("--radius-mm", type=float, default=30.0)
    parser.add_argument("--release-penalty", type=float, default=1e-3)
    parser.add_argument("--latent-dimensions", type=int, default=16)
    parser.add_argument("--train-count", type=int, default=0, help="0 uses every non-test frame")
    parser.add_argument("--test-stride", type=int, default=5)
    parser.add_argument("--test-offset", type=int, default=2)
    parser.add_argument(
        "--tracking-lag-frames",
        type=int,
        default=0,
        help="Use tracking from t+lag for target frame t; RGB/alpha and evaluation regions stay at t.",
    )
    parser.add_argument(
        "--correspondence-shuffle-fraction",
        type=float,
        default=0.0,
        help="Fraction of Gaussian attributes reassigned per frame while geometry stays fixed.",
    )
    parser.add_argument(
        "--correspondence-shuffle-mode",
        choices=("consistent", "local", "global"),
        default="consistent",
    )
    parser.add_argument("--correspondence-local-bin-mm", type=float, default=10.0)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--position-lr", type=float, default=2e-4)
    parser.add_argument("--max-scale-mm", type=float, default=5.0)
    parser.add_argument("--scale-penalty-threshold-mm", type=float, default=3.0)
    parser.add_argument("--scale-penalty-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.method in ("bounded", "adaptive") and args.radius_mm <= 0:
        parser.error("bounded/adaptive methods require --radius-mm > 0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    sequence_root = args.subject_root / "sequences" / args.sequence
    tracking_path = sequence_root / "tracking" / "flame2023_tracking_v2.npz"
    tracking = np.load(tracking_path)
    frame_count = len(tracking["expression"])
    video_path = sequence_root / "images" / "cam_222200037.mp4"
    alpha_path = sequence_root / "alpha_maps" / "cam_222200037.mp4"
    calibration = json.loads((args.subject_root / "calibration" / "camera_params.json").read_text())
    camera_id = "222200037"
    source = cv2.VideoCapture(str(video_path))
    source_height = float(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    source.release()
    if source_frames != frame_count:
        raise RuntimeError(f"tracking/video frame mismatch: {frame_count} vs {source_frames}")
    intrinsic = np.asarray(calibration["intrinsics"][camera_id], np.float32)
    intrinsic[:2] *= args.height / source_height
    viewmat = np.asarray(calibration["world_2_cam"][camera_id], np.float32)

    test_indices = np.arange(args.test_offset, frame_count, args.test_stride, dtype=np.int64)
    train_pool = np.setdiff1d(np.arange(frame_count, dtype=np.int64), test_indices)
    train_indices = nested_uniform_subset(train_pool, args.train_count)
    evaluation_indices = np.unique(np.r_[train_indices, test_indices])
    images_by_index = {
        int(index): read_video_frame(video_path, int(index), args.height).astype(np.float32) / 255.0
        for index in evaluation_indices
    }
    alpha_by_index = {
        int(index): read_video_frame(alpha_path, int(index), args.height)[..., [0]].astype(np.float32) / 255.0
        for index in evaluation_indices
    }
    images = np.stack([images_by_index[int(index)] for index in evaluation_indices])
    alpha = np.stack([alpha_by_index[int(index)] for index in evaluation_indices])
    index_to_loaded = {int(index): position for position, index in enumerate(evaluation_indices)}

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
        clean_hard_local = apply_dynamic_flame(assets, shape, expression, neck, jaw, eyes)
        zeros_expression = torch.zeros_like(expression[:1])
        neutral_local = apply_dynamic_flame(
            assets,
            shape,
            zeros_expression,
            torch.zeros_like(neck[:1]),
            torch.zeros_like(jaw[:1]),
            torch.zeros_like(eyes[:1]),
        )[0]
        clean_rotations = torch.from_numpy(tracking["rotation_matrices"]).float().to(device)
        clean_translations = torch.from_numpy(tracking["translation"]).float().to(device)
        scale = torch.from_numpy(tracking["scale"]).float().to(device)
        clean_hard_world = apply_tracking_world_transform(
            clean_hard_local, clean_rotations, clean_translations, scale
        )

    tracking_source_indices = np.clip(
        np.arange(frame_count, dtype=np.int64) + args.tracking_lag_frames,
        0,
        frame_count - 1,
    )
    tracking_source_t = torch.from_numpy(tracking_source_indices).long().to(device)
    hard_local = clean_hard_local[tracking_source_t]
    rotations = clean_rotations[tracking_source_t]
    translations = clean_translations[tracking_source_t]
    hard_world = apply_tracking_world_transform(hard_local, rotations, translations, scale)

    codes_np, code_report = tracking_codes(tracking, args.latent_dimensions)
    codes_np = codes_np[tracking_source_indices]
    codes = torch.from_numpy(codes_np).to(device)
    hard_world_np = hard_world.detach().cpu().numpy()
    clean_hard_world_np = clean_hard_world.detach().cpu().numpy()
    correspondence_np, effective_shuffle_fraction = build_correspondence_permutations(
        neutral_local.detach().cpu().numpy(),
        frame_count,
        args.correspondence_shuffle_fraction,
        args.correspondence_shuffle_mode,
        args.correspondence_local_bin_mm,
        args.seed,
    )
    correspondence = torch.from_numpy(correspondence_np).long().to(device)
    height, width = images.shape[1:3]
    semantic_masks = build_semantic_masks(
        clean_hard_world_np[evaluation_indices], linear_np["masks"], viewmat, intrinsic, height, width
    )
    loaded_train = np.asarray([index_to_loaded[int(index)] for index in train_indices])
    initial_colors = sample_initial_colors(
        images[loaded_train], alpha[loaded_train], hard_world_np[train_indices], viewmat, intrinsic
    )

    vertex_count = hard_local.shape[1]
    deformation_basis = torch.nn.Parameter(
        torch.zeros((args.latent_dimensions + 1, vertex_count, 3), device=device)
    )
    release_logits = torch.nn.Parameter(torch.full((vertex_count,), -3.0, device=device))
    color_logits = torch.nn.Parameter(torch.from_numpy(logit(initial_colors)).to(device))
    log_scale = torch.nn.Parameter(torch.full((vertex_count, 1), math.log(0.002), device=device))
    opacity_logits = torch.nn.Parameter(torch.zeros(vertex_count, device=device))
    quaternions = torch.zeros((vertex_count, 4), device=device)
    quaternions[:, 0] = 1.0
    parameter_groups = [
        {"params": [deformation_basis], "lr": args.position_lr / math.sqrt(args.latent_dimensions + 1)},
        {"params": [release_logits], "lr": 1e-2},
        {"params": [color_logits], "lr": 1e-2},
        {"params": [log_scale], "lr": 3e-3},
        {"params": [opacity_logits], "lr": 1e-2},
    ]
    optimizer = torch.optim.Adam(parameter_groups)
    viewmat_t = torch.from_numpy(viewmat).to(device)
    intrinsic_t = torch.from_numpy(intrinsic).to(device)
    images_t = torch.from_numpy(images).to(device)
    alpha_t = torch.from_numpy(alpha).to(device)
    head_masks_t = torch.from_numpy(semantic_masks["head"]).to(device)
    generator = np.random.default_rng(args.seed)
    history = []

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        batch = generator.choice(train_indices, min(args.batch_size, len(train_indices)), replace=False)
        loaded = torch.tensor([index_to_loaded[int(index)] for index in batch], device=device)
        batch_t = torch.from_numpy(batch).long().to(device)
        local, raw, gates = local_means_for_method(
            args.method,
            hard_local[batch_t],
            neutral_local,
            codes[batch_t],
            deformation_basis,
            release_logits,
            args.radius_mm,
        )
        world = apply_tracking_world_transform(local, rotations[batch_t], translations[batch_t], scale)
        scales = torch.exp(log_scale).clamp(5e-5, args.max_scale_mm / 1000.0).expand(-1, 3)
        colors = torch.sigmoid(color_logits)
        opacities = torch.sigmoid(opacity_logits)
        rendered, rendered_alpha = render_frames(
            world,
            quaternions,
            scales,
            opacities,
            colors,
            viewmat_t,
            intrinsic_t,
            width,
            height,
            correspondence[batch_t],
        )
        masks_t = head_masks_t[loaded]
        rgb_loss = (torch.abs(rendered - images_t[loaded]) * masks_t).sum() / (masks_t.sum() * 3).clamp_min(1)
        alpha_loss = (torch.abs(rendered_alpha - alpha_t[loaded]) * masks_t).sum() / masks_t.sum().clamp_min(1)
        scale_penalty = (
            torch.relu(scales - args.scale_penalty_threshold_mm / 1000.0).square().mean()
            * args.scale_penalty_weight
        )
        release_penalty = (
            args.release_penalty * gates.mean() if args.method == "adaptive" else 0.0 * gates.mean()
        )
        loss = rgb_loss + 0.2 * alpha_loss + scale_penalty + release_penalty
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            record = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "rgb_l1": float(rgb_loss.detach()),
                "alpha_l1": float(alpha_loss.detach()),
                "mean_scale_mm": float(scales.detach().mean() * 1000),
                "mean_raw_offset_mm": float(raw.detach().norm(dim=-1).mean() * 1000),
                "mean_release_gate": float(gates.detach().mean()),
            }
            history.append(record)
            print(json.dumps(record))

    evaluation = {}
    rendered_by_index: dict[int, np.ndarray] = {}
    alpha_rendered_by_index: dict[int, np.ndarray] = {}
    offset_norms = []
    with torch.no_grad():
        scales = torch.exp(log_scale).clamp(5e-5, args.max_scale_mm / 1000.0).expand(-1, 3)
        colors = torch.sigmoid(color_logits)
        opacities = torch.sigmoid(opacity_logits)
        for start in range(0, len(evaluation_indices), args.batch_size):
            frame_ids = evaluation_indices[start : start + args.batch_size]
            frame_t = torch.from_numpy(frame_ids).long().to(device)
            local, raw, gates = local_means_for_method(
                args.method,
                hard_local[frame_t],
                neutral_local,
                codes[frame_t],
                deformation_basis,
                release_logits,
                args.radius_mm,
            )
            world = apply_tracking_world_transform(local, rotations[frame_t], translations[frame_t], scale)
            rendered, rendered_alpha = render_frames(
                world,
                quaternions,
                scales,
                opacities,
                colors,
                viewmat_t,
                intrinsic_t,
                width,
                height,
                correspondence[frame_t],
            )
            offset_norms.extend((local - hard_local[frame_t]).norm(dim=-1).cpu().numpy())
            for frame_id, rgb, a in zip(frame_ids, rendered.cpu().numpy(), rendered_alpha.cpu().numpy()):
                rendered_by_index[int(frame_id)] = rgb
                alpha_rendered_by_index[int(frame_id)] = a

    split_indices = {"train": train_indices, "heldout": test_indices}
    for region_name, region_masks in semantic_masks.items():
        per_frame = []
        for frame_id in evaluation_indices:
            loaded = index_to_loaded[int(frame_id)]
            mask = region_masks[loaded]
            metrics = {
                "frame": int(frame_id),
                **masked_image_metrics(rendered_by_index[int(frame_id)], images[loaded], mask),
                **alpha_metrics(alpha_rendered_by_index[int(frame_id)] * mask, alpha[loaded] * mask),
            }
            per_frame.append(metrics)
        by_frame = {row["frame"]: row for row in per_frame}
        evaluation[region_name] = {"per_frame": per_frame}
        for split_name, split in split_indices.items():
            rows = [by_frame[int(index)] for index in split]
            evaluation[region_name][split_name] = {
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in ("masked_mse", "masked_mae", "masked_psnr", "alpha_mae", "alpha_iou")
            }

    args.output.mkdir(parents=True, exist_ok=True)
    montage = []
    qualitative_frames = nested_uniform_subset(test_indices, min(8, len(test_indices)))
    for frame_id in qualitative_frames:
        loaded = index_to_loaded[int(frame_id)]
        montage.extend([images[loaded], rendered_by_index[int(frame_id)]])
        save_rgb(args.output / f"heldout_{int(frame_id):04d}.png", rendered_by_index[int(frame_id)])
    save_montage(args.output / "heldout_target_render_montage.png", montage, columns=4)
    gate_values = torch.sigmoid(release_logits).detach().cpu().numpy()
    offset_values = np.concatenate(offset_norms, axis=0) * 1000.0
    np.savez_compressed(
        args.output / "model.npz",
        deformation_basis=deformation_basis.detach().cpu().numpy(),
        release_gates=gate_values,
        colors=colors.cpu().numpy(),
        scales=scales.cpu().numpy(),
        opacities=opacities.cpu().numpy(),
        train_indices=train_indices,
        heldout_indices=test_indices,
    )
    report = {
        "kind": "animation_bias_oracle",
        "causal_contract": {
            "fixed": ["G canonical FLAME queries", "C vertex correspondence", "P official tracking v2", "renderer"],
            "intervention": "A non-rigid animation mapping only",
            "hard": "tracked FLAME expression plus neck/jaw/eye LBS",
            "bounded": "hard A plus uniformly bounded learned residual",
            "adaptive": "hard A plus spatially gated bounded residual",
            "free": "neutral identity mesh plus learned tracking-code-conditioned deformation",
        },
        "subject_root": str(args.subject_root),
        "sequence": args.sequence,
        "tracking": str(tracking_path),
        "tracking_lag_frames": args.tracking_lag_frames,
        "tracking_lag_seconds_at_video_fps": float(args.tracking_lag_frames / 24.3),
        "tracking_source_boundary_rule": "clip",
        "correspondence_intervention": {
            "mode": args.correspondence_shuffle_mode,
            "requested_shuffle_fraction": args.correspondence_shuffle_fraction,
            "effective_changed_fraction": effective_shuffle_fraction,
            "local_bin_mm": args.correspondence_local_bin_mm if args.correspondence_shuffle_mode == "local" else None,
            "geometry_centers_changed": False,
            "per_frame_attribute_assignment": True,
        },
        "method": args.method,
        "radius_mm": args.radius_mm if args.method in ("bounded", "adaptive") else None,
        "release_penalty": args.release_penalty if args.method == "adaptive" else None,
        "train_count": int(len(train_indices)),
        "train_indices": train_indices.tolist(),
        "heldout_count": int(len(test_indices)),
        "heldout_indices": test_indices.tolist(),
        "temporal_split": {"stride": args.test_stride, "offset": args.test_offset},
        "tracking_code": code_report,
        "latent_dimensions": args.latent_dimensions,
        "steps": args.steps,
        "seed": args.seed,
        "resolution": [height, width],
        "gaussian_count": vertex_count,
        "max_scale_mm": args.max_scale_mm,
        "active_deformation_parameters": 0 if args.method == "hard" else int(deformation_basis.numel()),
        "total_parameter_count": int(
            deformation_basis.numel() + release_logits.numel() + color_logits.numel() + log_scale.numel() + opacity_logits.numel()
        ),
        "geometry": {
            "offset_from_hard_mean_mm": float(offset_values.mean()),
            "offset_from_hard_p90_mm": float(np.percentile(offset_values, 90)),
            "release_gate_mean": float(gate_values.mean()),
            "release_gate_p90": float(np.percentile(gate_values, 90)),
        },
        "history": history,
        "evaluation": evaluation,
        "limitations": [
            "The public training data contain one RGB camera, so held-out evaluation is temporal, not novel-view.",
            "Official tracking codes are available for held-out frames; this isolates A but is not a tracking-free deployment test.",
            "PCA of tracking codes uses no RGB/alpha targets but is transductive over the sequence.",
        ],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    compact = {
        region: {split: values[split] for split in ("train", "heldout")}
        for region, values in evaluation.items()
    }
    print(json.dumps({"output": str(args.output), "evaluation": compact, "geometry": report["geometry"]}, indent=2))


if __name__ == "__main__":
    main()
