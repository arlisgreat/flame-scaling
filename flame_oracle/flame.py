from __future__ import annotations

import glob
import inspect
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _enable_legacy_chumpy_import() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    try:
        import chumpy  # noqa: F401
        return
    except ImportError:
        pass
    candidates = []
    if os.environ.get("CHUMPY_SOURCE"):
        candidates.append(os.environ["CHUMPY_SOURCE"])
    candidates.extend(glob.glob("/home/airulan/.cache/uv/sdists-v9/pypi/chumpy/0.70/*/src"))
    for candidate in candidates:
        sys.path.insert(0, candidate)
        try:
            import chumpy  # noqa: F401
            return
        except ImportError:
            continue
    raise RuntimeError("FLAME pickle requires chumpy 0.70; install it or set CHUMPY_SOURCE")


def load_flame_linear_assets(
    model_path: Path,
    masks_path: Path,
    shape_dimensions: int = 100,
    expression_dimensions: int = 50,
) -> dict[str, object]:
    _enable_legacy_chumpy_import()
    with model_path.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    with masks_path.open("rb") as handle:
        masks = pickle.load(handle, encoding="latin1")
    shapedirs_object = model["shapedirs"]
    shapedirs = shapedirs_object.r if hasattr(shapedirs_object, "r") else np.asarray(shapedirs_object)
    directions = np.concatenate(
        [
            shapedirs[:, :, :shape_dimensions],
            shapedirs[:, :, 300 : 300 + expression_dimensions],
        ],
        axis=2,
    ).astype(np.float32)
    return {
        "template": np.asarray(model["v_template"], dtype=np.float32),
        "directions": directions,
        "faces": np.asarray(model["f"], dtype=np.int64),
        "masks": {key: np.unique(np.asarray(value, dtype=np.int64)) for key, value in masks.items()},
        "shape_dimensions": shape_dimensions,
        "expression_dimensions": expression_dimensions,
    }


def _as_numpy(value: object, dtype: np.dtype | type = np.float32) -> np.ndarray:
    if hasattr(value, "r"):
        value = value.r
    if "scipy.sparse" in str(type(value)):
        value = value.todense()
    return np.asarray(value, dtype=dtype)


def load_flame_dynamic_assets(
    model_path: Path,
    shape_dimensions: int = 300,
    expression_dimensions: int = 100,
) -> dict[str, np.ndarray | int]:
    """Load the minimal FLAME 2023 tensors required for differentiable LBS.

    This intentionally avoids the heavyweight PyTorch3D dependency used by
    VHAP/LAM.  The tensor slicing matches their FLAME implementation: shape
    directions start at zero and expression directions start at index 300.
    """
    _enable_legacy_chumpy_import()
    with model_path.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    shapedirs = _as_numpy(model["shapedirs"])
    shapedirs = np.concatenate(
        [
            shapedirs[:, :, :shape_dimensions],
            shapedirs[:, :, 300 : 300 + expression_dimensions],
        ],
        axis=2,
    ).astype(np.float32)
    posedirs = _as_numpy(model["posedirs"])
    posedirs = posedirs.reshape(-1, posedirs.shape[-1]).T.astype(np.float32)
    parents = _as_numpy(model["kintree_table"][0], np.int64).copy()
    parents[0] = -1
    return {
        "template": _as_numpy(model["v_template"]),
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "joint_regressor": _as_numpy(model["J_regressor"]),
        "parents": parents,
        "lbs_weights": _as_numpy(model["weights"]),
        "faces": _as_numpy(model["f"], np.int64),
        "shape_dimensions": shape_dimensions,
        "expression_dimensions": expression_dimensions,
    }


def batch_rodrigues(rotvecs: torch.Tensor) -> torch.Tensor:
    """Convert a batch of axis-angle vectors to rotation matrices."""
    if rotvecs.ndim != 2 or rotvecs.shape[-1] != 3:
        raise ValueError(f"expected [N,3] rotation vectors, got {tuple(rotvecs.shape)}")
    # Use the unnormalized skew matrix and sinc/Taylor coefficients.  The
    # normalized-axis form has a zero/undefined derivative at exactly zero,
    # which silently freezes jaw/neck/eye pose parameters initialized at rest.
    x, y, z = rotvecs.unbind(dim=1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=1
    ).reshape(-1, 3, 3)
    identity = torch.eye(3, dtype=rotvecs.dtype, device=rotvecs.device)[None]
    angle2 = torch.sum(rotvecs.square(), dim=1, keepdim=True)
    angle = torch.sqrt(angle2.clamp_min(1e-16))
    angle4 = angle2.square()
    small = angle2 < 1e-8
    coefficient_a = torch.where(
        small,
        1.0 - angle2 / 6.0 + angle4 / 120.0,
        torch.sin(angle) / angle,
    )[:, :, None]
    coefficient_b = torch.where(
        small,
        0.5 - angle2 / 24.0 + angle4 / 720.0,
        (1.0 - torch.cos(angle)) / angle2.clamp_min(1e-16),
    )[:, :, None]
    return identity + coefficient_a * skew + coefficient_b * torch.bmm(skew, skew)


def _transform_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [F.pad(rotation, [0, 0, 0, 1]), F.pad(translation, [0, 0, 0, 1], value=1.0)],
        dim=2,
    )


def _batch_rigid_transform(
    rotations: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    joints = joints.unsqueeze(-1)
    relative_joints = joints.clone()
    relative_joints[:, 1:] -= joints[:, parents[1:]]
    transforms = _transform_matrix(rotations.reshape(-1, 3, 3), relative_joints.reshape(-1, 3, 1))
    transforms = transforms.reshape(rotations.shape[0], rotations.shape[1], 4, 4)
    chain = [transforms[:, 0]]
    for joint_index in range(1, parents.shape[0]):
        chain.append(torch.matmul(chain[int(parents[joint_index])], transforms[:, joint_index]))
    chain = torch.stack(chain, dim=1)
    posed_joints = chain[:, :, :3, 3]
    relative = F.pad(torch.matmul(chain[:, :, :3, :3], joints), [0, 0, 0, 1])
    return posed_joints, chain - relative


def apply_dynamic_flame(
    assets: dict[str, torch.Tensor],
    shape: torch.Tensor,
    expression: torch.Tensor,
    neck: torch.Tensor,
    jaw: torch.Tensor,
    eyes: torch.Tensor,
) -> torch.Tensor:
    """Evaluate FLAME shape/blendshapes and LBS in model coordinates.

    Global head rotation, scale, and translation are deliberately excluded;
    use :func:`apply_tracking_world_transform` afterward to reproduce the
    NeRSemble benchmark provider's separated transformation convention.
    """
    batch_size = expression.shape[0]
    if shape.shape[0] == 1 and batch_size != 1:
        shape = shape.expand(batch_size, -1)
    coefficients = torch.cat([shape, expression], dim=1)
    template = assets["template"][None].expand(batch_size, -1, -1)
    vertices_shaped = template + torch.einsum("bl,vcl->bvc", coefficients, assets["shapedirs"])
    joints = torch.einsum("jv,bvc->bjc", assets["joint_regressor"], vertices_shaped)

    root = torch.zeros((batch_size, 3), dtype=expression.dtype, device=expression.device)
    full_pose = torch.cat([root, neck, jaw, eyes], dim=1)
    rotations = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
    identity = torch.eye(3, dtype=expression.dtype, device=expression.device)
    pose_feature = (rotations[:, 1:] - identity).reshape(batch_size, -1)
    pose_offsets = torch.matmul(pose_feature, assets["posedirs"]).reshape(batch_size, -1, 3)
    vertices_posed = vertices_shaped + pose_offsets

    _, transforms = _batch_rigid_transform(rotations, joints, assets["parents"])
    weights = assets["lbs_weights"][None].expand(batch_size, -1, -1)
    transforms = torch.matmul(weights, transforms.reshape(batch_size, transforms.shape[1], 16))
    transforms = transforms.reshape(batch_size, -1, 4, 4)
    homogeneous = torch.cat(
        [vertices_posed, torch.ones_like(vertices_posed[:, :, :1])], dim=2
    )
    return torch.matmul(transforms, homogeneous.unsqueeze(-1))[:, :, :3, 0]


def apply_tracking_world_transform(
    vertices: torch.Tensor,
    rotation_matrices: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply NeRSemble tracking's separated model-to-world transform."""
    if scale.numel() == 1:
        scale = scale.reshape(1, 1, 1)
    elif scale.ndim == 1:
        scale = scale[:, None, None]
    elif scale.ndim == 2:
        scale = scale[:, :, None]
    return scale * torch.matmul(vertices, rotation_matrices.transpose(1, 2)) + translation[:, None]


def rotation_matrix_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    x, y, z = rotvec.unbind()
    zero = torch.zeros((), dtype=rotvec.dtype, device=rotvec.device)
    skew = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    return torch.matrix_exp(skew)


def apply_linear_flame(
    template: torch.Tensor,
    directions: torch.Tensor,
    coefficients: torch.Tensor,
    rotvec: torch.Tensor,
    log_scale: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    shaped = template + torch.einsum("vcd,d->vc", directions, coefficients)
    rotation = rotation_matrix_from_rotvec(rotvec)
    return torch.exp(log_scale) * (shaped @ rotation.T) + translation
