#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def benchmark_subjects(root: Path, benchmark: str) -> list[str]:
    directory = root / benchmark
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


def stable_order(subjects: list[str], seed: int) -> list[str]:
    return sorted(
        subjects,
        key=lambda subject: hashlib.sha256(f"{seed}:{subject}".encode()).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("/data1/workspace/airulan/3dv/datasets/nersemble_v2"),
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/data1/workspace/airulan/3dv/datasets/nersemble_benchmark"),
    )
    parser.add_argument("--sequence", default="FREE")
    parser.add_argument("--source-camera", default="222200037")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 32, 128, 384])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_subjects = sorted(path.name for path in args.v2_root.iterdir() if path.is_dir())
    svfr_test = benchmark_subjects(args.benchmark_root, "svfr")
    nvs_external = benchmark_subjects(args.benchmark_root, "nvs")
    mono_external = benchmark_subjects(args.benchmark_root, "mono_flame_avatar")
    benchmark_union = sorted(set(svfr_test) | set(nvs_external) | set(mono_external))

    per_subject: dict[str, object] = {}
    usable_subjects: list[str] = []
    common_cameras: set[str] | None = None
    for subject in all_subjects:
        subject_root = args.v2_root / subject
        images_root = subject_root / "sequences" / args.sequence / "images"
        camera_files = sorted(images_root.glob("cam_*.mp4"))
        cameras = {path.stem.removeprefix("cam_") for path in camera_files}
        source_path = images_root / f"cam_{args.source_camera}.mp4"
        calibration_path = subject_root / "calibration" / "camera_params.json"
        background_path = (
            subject_root / "sequences" / "BACKGROUND" / f"image_{args.source_camera}.jpg"
        )
        source_frames = 0
        source_readable = False
        if source_path.exists():
            capture = cv2.VideoCapture(str(source_path))
            source_readable = capture.isOpened()
            source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if source_readable else 0
            capture.release()
        usable = bool(
            cameras
            and args.source_camera in cameras
            and calibration_path.exists()
            and background_path.exists()
            and source_readable
            and source_frames > 0
        )
        if usable:
            usable_subjects.append(subject)
            common_cameras = cameras if common_cameras is None else common_cameras & cameras
        per_subject[subject] = {
            "usable": usable,
            "camera_count": len(cameras),
            "camera_ids": sorted(cameras),
            "source_frame_count": source_frames,
            "source_video": str(source_path),
            "calibration": str(calibration_path),
            "source_background": str(background_path),
        }

    eligible_train = sorted(set(usable_subjects) - set(benchmark_union))
    ordered_train = stable_order(eligible_train, args.seed)
    invalid_levels = [level for level in args.levels if level > len(ordered_train)]
    if invalid_levels:
        parser.error(
            f"levels {invalid_levels} exceed {len(ordered_train)} eligible non-benchmark identities"
        )
    nested_train_sets = {str(level): ordered_train[:level] for level in args.levels}
    validation_subjects = ordered_train[max(args.levels) :]
    usable_svfr_test = sorted(set(svfr_test) & set(usable_subjects))
    common_camera_list = sorted(common_cameras or [])
    report = {
        "kind": "nersemble_v2_identity_scaling_audit",
        "v2_root": str(args.v2_root),
        "benchmark_root": str(args.benchmark_root),
        "sequence": args.sequence,
        "source_camera": args.source_camera,
        "seed": args.seed,
        "all_v2_subject_count": len(all_subjects),
        "usable_subject_count": len(usable_subjects),
        "benchmark_subjects": {
            "svfr_test": svfr_test,
            "nvs_external": nvs_external,
            "mono_external": mono_external,
            "union_count": len(benchmark_union),
        },
        "eligible_nonbenchmark_train_count": len(eligible_train),
        "usable_svfr_test_count": len(usable_svfr_test),
        "usable_svfr_test": usable_svfr_test,
        "common_camera_count": len(common_camera_list),
        "common_camera_ids": common_camera_list,
        "nested_train_sets": nested_train_sets,
        "validation_subjects": validation_subjects,
        "validation_subject_count": len(validation_subjects),
        "nested_contract": all(
            set(nested_train_sets[str(left)]).issubset(nested_train_sets[str(right)])
            for left, right in zip(args.levels[:-1], args.levels[1:])
        ),
        "per_subject": per_subject,
        "scope_warning": (
            "This manifest only establishes a leakage-free identity axis. It does not provide FLAME "
            "tracking or constitute a model result. SVFR identities are held out from every train level; "
            "NVS and mono benchmark identities are additionally excluded for external validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "all_v2_subject_count",
                    "usable_subject_count",
                    "eligible_nonbenchmark_train_count",
                    "usable_svfr_test_count",
                    "common_camera_count",
                    "common_camera_ids",
                    "nested_contract",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
