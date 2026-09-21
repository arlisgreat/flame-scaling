#!/usr/bin/env python3
"""Estimate paired hard-vs-free regional crossover from a tidy result ledger.

Positive delta always means the free method is better. The input must contain
method, n_id, capacity, region, seed, metric, value, plus an identity cluster
column (`identity_group` or `identity_id_hash`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else float("nan")


def first_crossover(points: dict[int, float]) -> dict[str, object]:
    scales = sorted(points)
    if not scales:
        return {"status": "no_data"}
    if points[scales[0]] >= 0:
        return {"status": "left_censored", "at_or_below": scales[0]}
    for left, right in zip(scales, scales[1:]):
        y_left, y_right = points[left], points[right]
        if y_left < 0 <= y_right:
            if y_right == y_left:
                estimate = float(right)
            else:
                fraction = -y_left / (y_right - y_left)
                estimate = math.exp(math.log(left) + fraction * (math.log(right) - math.log(left)))
            return {
                "status": "observed",
                "bracket": [left, right],
                "log_interpolated_estimate": estimate,
            }
    if all(points[scale] < 0 for scale in scales):
        return {"status": "right_censored", "above": scales[-1]}
    return {"status": "non_monotonic_no_negative_to_positive_crossing"}


def load_paired_differences(
    path: Path,
    metric: str,
    hard_method: str,
    free_method: str,
    lower_is_better: bool,
) -> tuple[list[dict[str, object]], str]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("metric") == metric]
    if not rows:
        raise ValueError(f"no rows found for metric={metric!r}")

    identity_column = "identity_group" if "identity_group" in rows[0] else "identity_id_hash"
    required = {"method", "n_id", "capacity", "region", "seed", "value", identity_column}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    # Average accidental repeated rows before pairing; frames must not become
    # independent samples in the statistical test.
    by_cell: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] not in {hard_method, free_method}:
            continue
        key = (
            row["method"],
            row["n_id"],
            row.get("n_obs", ""),
            row["capacity"],
            row["region"],
            row["seed"],
            row[identity_column],
            row.get("yaw_bin", "all"),
            row.get("sequence_family", "all"),
        )
        by_cell[key].append(float(row["value"]))

    paired: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for key, values in by_cell.items():
        method, *pair_key = key
        paired[tuple(pair_key)][method] = mean(values)

    differences = []
    for key, methods in paired.items():
        if hard_method not in methods or free_method not in methods:
            continue
        n_id, n_obs, capacity, region, seed, identity, yaw_bin, sequence_family = key
        raw = methods[hard_method] - methods[free_method]
        delta = raw if lower_is_better else -raw
        differences.append(
            {
                "n_id": int(n_id),
                "n_obs": int(n_obs) if n_obs else None,
                "capacity": capacity,
                "region": region,
                "seed": seed,
                "identity": identity,
                "yaw_bin": yaw_bin,
                "sequence_family": sequence_family,
                "delta_free_advantage": delta,
            }
        )
    if not differences:
        raise ValueError("no paired hard/free rows were found")
    return differences, identity_column


def summarize_group(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_n: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_n[int(row["n_id"])].append(float(row["delta_free_advantage"]))
    result = []
    for n_id in sorted(by_n):
        values = by_n[n_id]
        result.append(
            {
                "n_id": n_id,
                "paired_units": len(values),
                "mean_delta_free_advantage": mean(values),
                "median_delta_free_advantage": statistics.median(values),
            }
        )
    return result


def bootstrap_crossover(
    rows: list[dict[str, object]],
    replicates: int,
    rng: random.Random,
) -> dict[str, object]:
    identities = sorted({str(row["identity"]) for row in rows})
    seeds = sorted({str(row["seed"]) for row in rows})
    if not identities or not seeds:
        return {"replicates": 0, "reason": "no clusters"}

    by_cluster: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    n_values = sorted({int(row["n_id"]) for row in rows})
    for row in rows:
        key = (str(row["identity"]), str(row["seed"]), int(row["n_id"]))
        by_cluster[key].append(float(row["delta_free_advantage"]))

    estimates: list[float] = []
    statuses: dict[str, int] = defaultdict(int)
    for _ in range(replicates):
        sampled_identities = rng.choices(identities, k=len(identities))
        sampled_seeds = rng.choices(seeds, k=len(seeds))
        points: dict[int, float] = {}
        for n_id in n_values:
            values = []
            for identity in sampled_identities:
                for seed in sampled_seeds:
                    values.extend(by_cluster.get((identity, seed, n_id), []))
            if values:
                points[n_id] = mean(values)
        crossing = first_crossover(points)
        status = str(crossing["status"])
        statuses[status] += 1
        if status == "observed":
            estimates.append(float(crossing["log_interpolated_estimate"]))

    return {
        "replicates": replicates,
        "status_fractions": {key: value / replicates for key, value in sorted(statuses.items())},
        "observed_crossover_count": len(estimates),
        "crossover_estimate_ci95_conditional_on_observed": [
            percentile(estimates, 0.025),
            percentile(estimates, 0.975),
        ],
        "note": "CI is conditional on an observed crossing; censoring fractions must be reported beside it.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--lower-is-better", action="store_true")
    direction.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--hard-method", default="hard")
    parser.add_argument("--free-method", default="free")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    differences, identity_column = load_paired_differences(
        args.input,
        args.metric,
        args.hard_method,
        args.free_method,
        args.lower_is_better,
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in differences:
        key = (
            str(row["capacity"]),
            str(row["region"]),
            str(row["yaw_bin"]),
            str(row["sequence_family"]),
        )
        grouped[key].append(row)

    rng = random.Random(args.random_seed)
    analyses = []
    for (capacity, region, yaw_bin, sequence_family), rows in sorted(grouped.items()):
        summary = summarize_group(rows)
        points = {int(item["n_id"]): float(item["mean_delta_free_advantage"]) for item in summary}
        analyses.append(
            {
                "capacity": capacity,
                "region": region,
                "yaw_bin": yaw_bin,
                "sequence_family": sequence_family,
                "by_identity_scale": summary,
                "observed_crossover": first_crossover(points),
                "cluster_bootstrap": bootstrap_crossover(rows, args.bootstrap, rng),
            }
        )

    report = {
        "metric": args.metric,
        "positive_delta_means": f"{args.free_method} is better than {args.hard_method}",
        "identity_cluster_column": identity_column,
        "paired_difference_rows": len(differences),
        "analyses": analyses,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

