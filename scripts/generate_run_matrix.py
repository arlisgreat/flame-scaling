#!/usr/bin/env python3
"""Generate a deterministic run ledger from configs/study.json."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


FIELDS = [
    "run_id",
    "stage",
    "split_id",
    "method",
    "g_state",
    "c_state",
    "a_state",
    "p_state",
    "n_id",
    "n_obs",
    "capacity",
    "seed",
    "status",
]


def stable_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "run_" + hashlib.sha256(encoded).hexdigest()[:12]


def rows_for_stage(
    stage: str,
    section: dict[str, object],
    seeds: list[int],
    states: dict[str, list[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    axes = itertools.product(
        section["identity_counts"],
        section["observations_per_identity"],
        section["capacities"],
        section["methods"],
        seeds,
    )
    for n_id, n_obs, capacity, method, seed in axes:
        role_state = states[str(method)]
        payload = {
            "stage": stage,
            "split_id": section["split_id"],
            "method": method,
            "n_id": n_id,
            "n_obs": n_obs,
            "capacity": capacity,
            "seed": seed,
        }
        rows.append(
            {
                "run_id": stable_id(payload),
                **payload,
                "g_state": role_state[0],
                "c_state": role_state[1],
                "a_state": role_state[2],
                "p_state": role_state[3],
                "status": "planned",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    seeds = config["paired_seeds"]
    states = config["role_states"]
    rows: list[dict[str, object]] = []
    for stage, key in (
        ("main", "main"),
        ("observation", "observation_scan"),
        ("role", "role_isolates"),
    ):
        rows.extend(rows_for_stage(stage, config[key], seeds, states))

    seen = set()
    for row in rows:
        if row["run_id"] in seen:
            raise RuntimeError(f"duplicate run id: {row['run_id']}")
        seen.add(row["run_id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {stage: sum(row["stage"] == stage for row in rows) for stage in ("main", "observation", "role")}
    print(json.dumps({"output": str(args.output), "total": len(rows), "by_stage": counts}, indent=2))


if __name__ == "__main__":
    main()

