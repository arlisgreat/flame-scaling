from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np


def read_binary_pcd(path: Path) -> dict[str, np.ndarray]:
    """Read the fixed NeRSemble binary PCD layout without Open3D."""
    payload = path.read_bytes()
    marker = b"DATA binary\n"
    if marker not in payload:
        raise ValueError(f"only binary PCD is supported: {path}")
    header, body = payload.split(marker, 1)
    header_text = header.decode("ascii", errors="strict")
    fields_match = re.search(r"^FIELDS (.+)$", header_text, flags=re.MULTILINE)
    points_match = re.search(r"^POINTS (\d+)$", header_text, flags=re.MULTILINE)
    if not fields_match or not points_match:
        raise ValueError(f"invalid PCD header: {path}")
    fields = fields_match.group(1).split()
    expected = ["x", "y", "z", "normal_x", "normal_y", "normal_z", "rgb"]
    if fields != expected:
        raise ValueError(f"unexpected PCD fields {fields}; expected {expected}")
    points = int(points_match.group(1))
    values = np.frombuffer(body, dtype=np.float32)
    if values.size != points * len(fields):
        raise ValueError(f"PCD size mismatch: header={points}, values={values.size}")
    values = values.reshape(points, len(fields))
    packed = values[:, 6].view(np.uint32)
    rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1).astype(np.uint8)
    return {
        "xyz": values[:, :3].copy(),
        "normals": values[:, 3:6].copy(),
        "rgb": rgb,
    }


def read_point_cloud(path: Path) -> dict[str, np.ndarray]:
    """Read either a NeRSemble binary PCD or an oracle-generated NPZ cloud."""
    if path.suffix.lower() == ".pcd":
        return read_binary_pcd(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"unsupported point-cloud format: {path}")
    payload = np.load(path)
    if "xyz" not in payload:
        raise ValueError(f"point-cloud NPZ is missing xyz: {path}")
    xyz = payload["xyz"].astype(np.float32)
    normals = payload["normals"].astype(np.float32) if "normals" in payload else np.zeros_like(xyz)
    rgb = payload["rgb"] if "rgb" in payload else np.full_like(xyz, 128, dtype=np.uint8)
    if rgb.dtype.kind == "f" and rgb.max(initial=0.0) <= 1.0:
        rgb = rgb * 255.0
    return {"xyz": xyz, "normals": normals, "rgb": np.clip(rgb, 0, 255).astype(np.uint8)}


def read_video_frame(path: Path, frame_index: int, output_height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = capture.read()
    capture.release()
    if not ok or bgr is None:
        raise RuntimeError(f"cannot decode frame {frame_index}: {path}")
    height, width = bgr.shape[:2]
    output_width = round(width * output_height / height)
    interpolation = cv2.INTER_AREA if output_height < height else cv2.INTER_LINEAR
    bgr = cv2.resize(bgr, (output_width, output_height), interpolation=interpolation)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_image(path: Path, output_height: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image: {path}")
    height, width = bgr.shape[:2]
    output_width = round(width * output_height / height)
    interpolation = cv2.INTER_AREA if output_height < height else cv2.INTER_LINEAR
    bgr = cv2.resize(bgr, (output_width, output_height), interpolation=interpolation)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def background_subtraction_alpha(
    rgb: np.ndarray,
    background: np.ndarray,
    threshold: int = 30,
) -> np.ndarray:
    if rgb.shape != background.shape:
        raise ValueError(f"RGB/background shape mismatch: {rgb.shape} vs {background.shape}")
    difference = np.max(np.abs(rgb.astype(np.int16) - background.astype(np.int16)), axis=-1).astype(np.uint8)
    mask = (difference >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise RuntimeError("background subtraction produced no foreground component")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return cv2.GaussianBlur(filled.astype(np.float32) / 255.0, (3, 3), 0.6)


def load_multiview_frame(
    subject_root: Path,
    sequence: str,
    frame_index: int,
    output_height: int,
    background_threshold: int = 30,
) -> dict[str, object]:
    sequence_root = subject_root / "sequences" / sequence
    calibration = json.loads((subject_root / "calibration" / "camera_params.json").read_text())
    rgb_paths = {
        path.stem.removeprefix("cam_"): path for path in (sequence_root / "images").glob("cam_*.mp4")
    }
    alpha_paths = {
        path.stem.removeprefix("cam_"): path for path in (sequence_root / "alpha_maps").glob("cam_*.mp4")
    }
    background_root = subject_root / "sequences" / "BACKGROUND"
    background_paths = {
        path.stem.removeprefix("image_"): path for path in background_root.glob("image_*.jpg")
    }
    camera_ids = [camera_id for camera_id in calibration["world_2_cam"] if camera_id in rgb_paths]
    alpha_source = "provided_video" if all(camera_id in alpha_paths for camera_id in camera_ids) else "background_subtraction"
    if alpha_source == "background_subtraction":
        camera_ids = [camera_id for camera_id in camera_ids if camera_id in background_paths]
    if not camera_ids:
        raise RuntimeError(f"no RGB/alpha camera intersection under {sequence_root}")

    images = []
    alpha = []
    intrinsics = []
    viewmats = []
    for camera_id in camera_ids:
        rgb = read_video_frame(rgb_paths[camera_id], frame_index, output_height)
        if alpha_source == "provided_video":
            a = read_video_frame(alpha_paths[camera_id], frame_index, output_height)[..., 0].astype(np.float32) / 255.0
        else:
            background = read_image(background_paths[camera_id], output_height)
            a = background_subtraction_alpha(rgb, background, threshold=background_threshold)
        images.append(rgb.astype(np.float32) / 255.0)
        alpha.append(a.astype(np.float32))

        metadata_capture = cv2.VideoCapture(str(rgb_paths[camera_id]))
        source_height = metadata_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        metadata_capture.release()
        if source_height <= 0:
            raise RuntimeError(f"invalid source height: {rgb_paths[camera_id]}")
        scale = output_height / source_height
        K_value = calibration["intrinsics"]
        K = np.asarray(K_value[camera_id] if isinstance(K_value, dict) else K_value, dtype=np.float32).copy()
        K[0, :] *= scale
        K[1, :] *= scale
        intrinsics.append(K)
        viewmats.append(np.asarray(calibration["world_2_cam"][camera_id], dtype=np.float32))

    return {
        "camera_ids": camera_ids,
        "images": np.stack(images),
        "alpha": np.stack(alpha)[..., None],
        "Ks": np.stack(intrinsics),
        "viewmats": np.stack(viewmats),
        "width": images[0].shape[1],
        "height": images[0].shape[0],
        "sequence_root": str(sequence_root),
        "alpha_source": alpha_source,
        "background_threshold": background_threshold if alpha_source == "background_subtraction" else None,
    }


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image * 255.0 if image.dtype.kind == "f" else image, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_montage(path: Path, images: list[np.ndarray], columns: int = 4) -> None:
    if not images:
        return
    materialized = [np.clip(image * 255.0, 0, 255).astype(np.uint8) if image.dtype.kind == "f" else image for image in images]
    height, width = materialized[0].shape[:2]
    blank = np.zeros_like(materialized[0])
    while len(materialized) % columns:
        materialized.append(blank)
    rows = [np.concatenate(materialized[i : i + columns], axis=1) for i in range(0, len(materialized), columns)]
    save_rgb(path, np.concatenate(rows, axis=0))
