#!/usr/bin/env python3
"""Read-only audit of local compute, NeRSemble metadata, and LAM assets."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


DEFAULT_ROOTS = {
    "nersemble_v2": Path("/data1/workspace/airulan/3dv/datasets/nersemble_v2"),
    "nersemble_benchmark": Path("/data1/workspace/airulan/3dv/datasets/nersemble_benchmark"),
    "lam": Path("/data1/workspace/airulan/3dv/external/LAM"),
}


def gpu_inventory() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in output.splitlines():
        index, name, total, used = [part.strip() for part in line.split(",", 3)]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
            }
        )
    return rows


def sequence_inventory(root: Path) -> dict[str, object]:
    path = root / "metadata_sequences.csv"
    if not path.is_file():
        return {"metadata_present": False}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sequence_columns = [key for key in rows[0] if key not in {"ID", "wears_glasses"}] if rows else []
    present = {
        column: sum(str(row.get(column, "")).strip().lower() in {"x", "m"} for row in rows)
        for column in sequence_columns
    }
    expected_directories = 0
    missing_directories = 0
    sequences_with_fewer_than_16_videos = 0
    video_files = 0
    background_images = 0
    for row in rows:
        participant = str(row["ID"]).zfill(3)
        for column in sequence_columns:
            if str(row.get(column, "")).strip().lower() not in {"x", "m"}:
                continue
            if column == "BACKGROUND":
                background_dir = root / participant / "sequences" / column
                background_images += sum(1 for _ in background_dir.glob("*.jpg"))
                continue
            expected_directories += 1
            image_dir = root / participant / "sequences" / column / "images"
            if not image_dir.is_dir():
                missing_directories += 1
                continue
            count = sum(1 for _ in image_dir.glob("*.mp4"))
            video_files += count
            if count < 16:
                sequences_with_fewer_than_16_videos += 1
    return {
        "metadata_present": True,
        "metadata_rows": len(rows),
        "available_by_sequence": present,
        "physical_video_check": {
            "expected_sequence_directories": expected_directories,
            "missing_sequence_directories": missing_directories,
            "sequence_directories_with_fewer_than_16_mp4": sequences_with_fewer_than_16_videos,
            "mp4_files_counted": video_files,
            "background_jpg_files_counted": background_images,
        },
    }


def lam_inventory(root: Path) -> dict[str, object]:
    runner_dir = root / "lam" / "runners"
    python_runners = sorted(path.name for path in runner_dir.glob("*.py")) if runner_dir.is_dir() else []
    training_entrypoints = sorted(
        str(path.relative_to(root))
        for path in root.glob("**/train*.py")
        if ".git" not in path.parts
    )
    return {
        "present": root.is_dir(),
        "license_present": (root / "LICENSE").is_file(),
        "runner_python_files": python_runners,
        "training_entrypoints": training_entrypoints,
        "note": "File presence is not proof that a released end-to-end training recipe is runnable.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nersemble = DEFAULT_ROOTS["nersemble_v2"]
    participant_dirs = []
    if nersemble.is_dir():
        participant_dirs = sorted(path.name for path in nersemble.iterdir() if path.is_dir() and path.name.isdigit())

    report = {
        "audit_kind": "read_only_local_inventory",
        "gpu": gpu_inventory(),
        "paths": {name: {"path": str(path), "present": path.exists()} for name, path in DEFAULT_ROOTS.items()},
        "nersemble_v2": {
            "participant_directory_count": len(participant_dirs),
            "first_participant_ids": participant_dirs[:10],
            **sequence_inventory(nersemble),
        },
        "lam": lam_inventory(DEFAULT_ROOTS["lam"]),
        "warnings": [
            "This audit does not validate licenses or consent terms.",
            "This audit does not decode every video or prove dataset completeness.",
            "Raw identifiable data must remain outside this repository.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "participants": len(participant_dirs), "gpus": len(report["gpu"])}, indent=2))


if __name__ == "__main__":
    main()
