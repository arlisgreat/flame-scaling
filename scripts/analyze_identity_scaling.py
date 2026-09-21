#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


RUN_PATTERN = re.compile(r"nid(?P<n_id>\d+)_(?P<method>hard|adaptive|released)_seed(?P<seed>\d+)")
METHODS = ["hard", "adaptive", "released"]
METHOD_LABELS = {"hard": "Hard FLAME", "adaptive": "Adaptive release", "released": "Released $G_\\mu$"}
METHOD_COLORS = {"hard": "#345995", "adaptive": "#E67E22", "released": "#2A9D8F"}


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(root.glob("*/metrics.json")):
        match = RUN_PATTERN.fullmatch(path.parent.name)
        if not match:
            continue
        payload = json.loads(path.read_text())
        runs.append(
            {
                "path": path,
                "n_id": int(match.group("n_id")),
                "method": match.group("method"),
                "seed": int(match.group("seed")),
                "payload": payload,
            }
        )
    return runs


def identity_metric_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    metrics = [
        "masked_psnr",
        "masked_mae",
        "masked_mse",
        "alpha_mae",
        "alpha_iou",
        "ssim",
        "lpips_alex_spatial",
    ]
    for run in runs:
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in run["payload"]["test_rows"]:
            grouped[(row["subject"], int(row["n_obs"]), row["region"])].append(row)
        for (subject, n_obs, region), camera_rows in grouped.items():
            output = {
                "n_id": run["n_id"],
                "method": run["method"],
                "seed": run["seed"],
                "subject": subject,
                "n_obs": n_obs,
                "region": region,
                "camera_count": len(camera_rows),
                "pixel_count_mean": float(np.mean([row["pixel_count"] for row in camera_rows])),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in camera_rows if metric in row]
                if values:
                    output[metric] = float(np.mean(values))
            rows.append(output)
    return rows


def geometry_identity_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for row in run["payload"]["geometry_rows"]:
            rows.append(
                {
                    "n_id": run["n_id"],
                    "method": run["method"],
                    "seed": run["seed"],
                    "subject": row["subject"],
                    "n_obs": int(row["n_obs"]),
                    **{f"all_{key}": value for key, value in row["all"].items()},
                    **{
                        f"{region}_{key}": value
                        for region, values in row["regions"].items()
                        for key, value in values.items()
                    },
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    rows: list[dict[str, Any]], metric: str, rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    per_identity: dict[tuple[int, str, int, str, str], list[float]] = defaultdict(list)
    cell_seeds: dict[tuple[int, str, int, str], set[int]] = defaultdict(set)
    for row in rows:
        if metric in row:
            cell = (row["n_id"], row["method"], row["n_obs"], row["region"])
            per_identity[(*cell, row["subject"])].append(row[metric])
            cell_seeds[cell].add(row["seed"])
    grouped: dict[tuple[int, str, int, str], list[float]] = defaultdict(list)
    for (*cell, _subject), values in per_identity.items():
        grouped[tuple(cell)].append(float(np.mean(values)))
    output = []
    for (n_id, method, n_obs, region), values in sorted(grouped.items()):
        mean, low, high = bootstrap_mean(np.asarray(values), rng, samples)
        output.append(
            {
                "n_id": n_id,
                "method": method,
                "n_obs": n_obs,
                "region": region,
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "identity_count": len(values),
                "seed_count": len(cell_seeds[(n_id, method, n_obs, region)]),
                "seed_aggregation": "mean within held-out identity before identity bootstrap",
            }
        )
    return output


def metric_advantage(comparison: float, hard: float, metric: str) -> float:
    return (
        hard - comparison
        if metric in {"masked_mae", "masked_mse", "alpha_mae", "lpips_alex_spatial"}
        else comparison - hard
    )


def paired_effects(
    rows: list[dict[str, Any]], metric: str, rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    raw_lookup: dict[tuple[int, str, int, str, str], list[float]] = defaultdict(list)
    cell_seeds: dict[tuple[int, int, str], set[int]] = defaultdict(set)
    for row in rows:
        if metric in row:
            raw_lookup[
                (row["n_id"], row["method"], row["n_obs"], row["region"], row["subject"])
            ].append(row[metric])
            cell_seeds[(row["n_id"], row["n_obs"], row["region"])].add(row["seed"])
    lookup = {key: float(np.mean(values)) for key, values in raw_lookup.items()}
    cells = sorted({(row["n_id"], row["n_obs"], row["region"]) for row in rows if metric in row})
    output = []
    for n_id, n_obs, region in cells:
        for comparison in ["adaptive", "released"]:
            subjects = sorted(
                {
                    key[4]
                    for key in lookup
                    if key[:4] == (n_id, "hard", n_obs, region)
                    and (n_id, comparison, n_obs, region, key[4]) in lookup
                }
            )
            values = np.asarray(
                [
                    metric_advantage(
                        lookup[(n_id, comparison, n_obs, region, subject)],
                        lookup[(n_id, "hard", n_obs, region, subject)],
                        metric,
                    )
                    for subject in subjects
                ]
            )
            mean, low, high = bootstrap_mean(values, rng, samples)
            output.append(
                {
                    "n_id": n_id,
                    "n_obs": n_obs,
                    "region": region,
                    "metric": metric,
                    "comparison": comparison,
                    "advantage_over_hard_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "identity_count": len(values),
                    "seed_count": len(cell_seeds[(n_id, n_obs, region)]),
                    "seed_aggregation": "paired effect averaged within identity across seeds",
                    "fraction_identities_advantage_positive": float(np.mean(values > 0)),
                }
            )
    return output


def scaling_interactions(
    rows: list[dict[str, Any]], metric: str, rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    levels = sorted({row["n_id"] for row in rows})
    if len(levels) < 2:
        return []
    low_n, high_n = levels[0], levels[-1]
    raw_lookup: dict[tuple[int, str, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if metric in row:
            raw_lookup[
                (row["n_id"], row["method"], row["n_obs"], row["region"], row["subject"])
            ].append(row[metric])
    lookup = {key: float(np.mean(values)) for key, values in raw_lookup.items()}
    cells = sorted({(row["n_obs"], row["region"]) for row in rows if metric in row})
    output = []
    for n_obs, region in cells:
        for comparison in ["adaptive", "released"]:
            subjects = sorted(
                set.intersection(
                    *[
                        {
                            key[4]
                            for key in lookup
                            if key[:4] == (n_id, method, n_obs, region)
                        }
                        for n_id in [low_n, high_n]
                        for method in ["hard", comparison]
                    ]
                )
            )
            values = []
            for subject in subjects:
                low_gap = metric_advantage(
                    lookup[(low_n, comparison, n_obs, region, subject)],
                    lookup[(low_n, "hard", n_obs, region, subject)],
                    metric,
                )
                high_gap = metric_advantage(
                    lookup[(high_n, comparison, n_obs, region, subject)],
                    lookup[(high_n, "hard", n_obs, region, subject)],
                    metric,
                )
                values.append(high_gap - low_gap)
            mean, low, high = bootstrap_mean(np.asarray(values), rng, samples)
            output.append(
                {
                    "low_n_id": low_n,
                    "high_n_id": high_n,
                    "n_obs": n_obs,
                    "region": region,
                    "metric": metric,
                    "comparison": comparison,
                    "release_advantage_change": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "identity_count": len(values),
                    "seed_aggregation": "paired gaps averaged within identity across seeds",
                }
            )
    return output


def crossover_status(effects: list[dict[str, Any]], comparison: str, n_obs: int, region: str) -> dict[str, Any]:
    points = sorted(
        [
            (row["n_id"], row["advantage_over_hard_mean"])
            for row in effects
            if row["comparison"] == comparison and row["n_obs"] == n_obs and row["region"] == region
        ]
    )
    if not points:
        return {"status": "missing"}
    if points[0][1] > 0:
        return {"status": "left_censored", "below": points[0][0], "points": points}
    for (left_n, left), (right_n, right) in zip(points[:-1], points[1:]):
        if left <= 0 < right:
            fraction = -left / max(right - left, 1e-12)
            estimate = math.exp(math.log(left_n) + fraction * (math.log(right_n) - math.log(left_n)))
            return {
                "status": "observed",
                "bracket": [left_n, right_n],
                "log_interpolated_estimate": estimate,
                "points": points,
            }
    return {"status": "right_censored", "above": points[-1][0], "points": points}


def plot_scaling(aggregates: list[dict[str, Any]], output: Path) -> None:
    regions = ["full_frame", "flame_unsupported", "oral"]
    observations = sorted({row["n_obs"] for row in aggregates})
    fig, axes = plt.subplots(len(regions), len(observations), figsize=(4.2 * len(observations), 3.4 * len(regions)), squeeze=False)
    for row_index, region in enumerate(regions):
        for column_index, n_obs in enumerate(observations):
            ax = axes[row_index, column_index]
            for method in METHODS:
                cells = sorted(
                    [row for row in aggregates if row["region"] == region and row["n_obs"] == n_obs and row["method"] == method],
                    key=lambda row: row["n_id"],
                )
                if not cells:
                    continue
                x = np.asarray([row["n_id"] for row in cells])
                y = np.asarray([row["mean"] for row in cells])
                low = np.asarray([row["ci_low"] for row in cells])
                high = np.asarray([row["ci_high"] for row in cells])
                ax.plot(x, y, marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
                ax.fill_between(x, low, high, color=METHOD_COLORS[method], alpha=0.15)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted({row["n_id"] for row in aggregates}))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.grid(alpha=0.2)
            ax.set_title(f"{region}; $N_{{obs}}={n_obs}$")
            ax.set_xlabel("Training identities")
            ax.set_ylabel("Identity-clustered PSNR (dB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output / "identity_scaling_psnr.png", dpi=220)
    plt.close(fig)


def plot_advantage_heatmaps(effects: list[dict[str, Any]], output: Path) -> None:
    regions = ["full_frame", "flame_unsupported", "oral"]
    comparisons = ["adaptive", "released"]
    n_ids = sorted({row["n_id"] for row in effects})
    observations = sorted({row["n_obs"] for row in effects})
    fig, axes = plt.subplots(len(regions), len(comparisons), figsize=(9, 9), squeeze=False)
    all_values = [abs(row["advantage_over_hard_mean"]) for row in effects if row["region"] in regions]
    limit = max(all_values, default=1.0)
    for i, region in enumerate(regions):
        for j, comparison in enumerate(comparisons):
            matrix = np.full((len(observations), len(n_ids)), np.nan)
            for row in effects:
                if row["region"] == region and row["comparison"] == comparison:
                    matrix[observations.index(row["n_obs"]), n_ids.index(row["n_id"])] = row["advantage_over_hard_mean"]
            ax = axes[i, j]
            image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
            for y in range(len(observations)):
                for x in range(len(n_ids)):
                    if np.isfinite(matrix[y, x]):
                        ax.text(x, y, f"{matrix[y, x]:+.2f}", ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(n_ids)), n_ids)
            ax.set_yticks(range(len(observations)), observations)
            ax.set_xlabel("Training identities")
            ax.set_ylabel("Observed views")
            ax.set_title(f"{region}: {comparison} - hard")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="PSNR advantage over hard (dB)", shrink=0.75)
    fig.savefig(output / "release_advantage_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(identity_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]], output: Path) -> None:
    quality = defaultdict(list)
    for row in identity_rows:
        if row["region"] == "full_frame" and row["n_obs"] == 4:
            quality[(row["n_id"], row["method"])].append(row["masked_psnr"])
    offsets = defaultdict(list)
    for row in geometry_rows:
        if row["n_obs"] == 4:
            offsets[(row["n_id"], row["method"])].append(row["all_offset_mean_mm"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for method in METHODS:
        for n_id in sorted({key[0] for key in quality}):
            key = (n_id, method)
            if key not in offsets or key not in quality:
                continue
            x = np.mean(offsets[key])
            y = np.mean(quality[key])
            ax.scatter(x, y, s=75, color=METHOD_COLORS[method], marker={"hard": "s", "adaptive": "o", "released": "^"}[method])
            ax.annotate(str(n_id), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlabel("Mean Gaussian displacement from FLAME (mm; lower is stronger prior)")
    ax.set_ylabel("Full-frame PSNR at $N_{obs}=4$ (dB)")
    handles = [
        plt.Line2D([0], [0], marker={"hard": "s", "adaptive": "o", "released": "^"}[method], color="none", markerfacecolor=METHOD_COLORS[method], label=METHOD_LABELS[method], markersize=8)
        for method in METHODS
    ]
    ax.legend(handles=handles, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "quality_release_pareto.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-audit",
        type=Path,
        help="Optional cache_audit.json; flagged identities are excluded only in a labeled sensitivity analysis.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--expected-training-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    expected = {
        (n_id, method, seed)
        for n_id in [8, 32, 128, 384]
        for method in METHODS
        for seed in args.expected_training_seeds
    }
    observed = {(run["n_id"], run["method"], run["seed"]) for run in runs}
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"identity scaling matrix is incomplete: {missing}")
    identity_rows = identity_metric_rows(runs)
    geometry_rows = geometry_identity_rows(runs)
    write_csv(args.output / "identity_metrics.csv", identity_rows)
    write_csv(args.output / "identity_geometry.csv", geometry_rows)
    rng = np.random.default_rng(args.seed)
    aggregate_psnr = aggregate_rows(identity_rows, "masked_psnr", rng, args.bootstrap_samples)
    effects_psnr = paired_effects(identity_rows, "masked_psnr", rng, args.bootstrap_samples)
    per_seed_effects_psnr = []
    for training_seed in args.expected_training_seeds:
        seed_rows = [row for row in identity_rows if row["seed"] == training_seed]
        for row in paired_effects(seed_rows, "masked_psnr", rng, args.bootstrap_samples):
            row["training_seed"] = training_seed
            per_seed_effects_psnr.append(row)
    interactions_psnr = scaling_interactions(identity_rows, "masked_psnr", rng, args.bootstrap_samples)
    aggregate_mae = aggregate_rows(identity_rows, "masked_mae", rng, args.bootstrap_samples)
    effects_mae = paired_effects(identity_rows, "masked_mae", rng, args.bootstrap_samples)
    aggregate_lpips = aggregate_rows(
        identity_rows, "lpips_alex_spatial", rng, args.bootstrap_samples
    )
    effects_lpips = paired_effects(
        identity_rows, "lpips_alex_spatial", rng, args.bootstrap_samples
    )
    interactions_lpips = scaling_interactions(
        identity_rows, "lpips_alex_spatial", rng, args.bootstrap_samples
    )
    flagged_subjects: set[str] = set()
    sensitivity_effects_psnr: list[dict[str, Any]] = []
    sensitivity_interactions_psnr: list[dict[str, Any]] = []
    if args.cache_audit is not None:
        audit = json.loads(args.cache_audit.read_text())
        flagged_subjects = {row["subject"] for row in audit["flagged_subjects"]}
        sensitivity_rows = [row for row in identity_rows if row["subject"] not in flagged_subjects]
        sensitivity_effects_psnr = paired_effects(
            sensitivity_rows, "masked_psnr", rng, args.bootstrap_samples
        )
        sensitivity_interactions_psnr = scaling_interactions(
            sensitivity_rows, "masked_psnr", rng, args.bootstrap_samples
        )
    write_csv(args.output / "aggregate_psnr.csv", aggregate_psnr)
    write_csv(args.output / "paired_psnr_effects.csv", effects_psnr)
    write_csv(args.output / "per_seed_paired_psnr_effects.csv", per_seed_effects_psnr)
    write_csv(args.output / "scaling_interactions_psnr.csv", interactions_psnr)
    write_csv(args.output / "aggregate_mae.csv", aggregate_mae)
    write_csv(args.output / "paired_mae_effects.csv", effects_mae)
    write_csv(args.output / "aggregate_lpips.csv", aggregate_lpips)
    write_csv(args.output / "paired_lpips_effects.csv", effects_lpips)
    write_csv(args.output / "scaling_interactions_lpips.csv", interactions_lpips)
    write_csv(args.output / "sensitivity_paired_psnr_effects.csv", sensitivity_effects_psnr)
    write_csv(
        args.output / "sensitivity_scaling_interactions_psnr.csv",
        sensitivity_interactions_psnr,
    )
    crossovers = {
        f"{comparison}_nobs{n_obs}_{region}": crossover_status(effects_psnr, comparison, n_obs, region)
        for comparison in ["adaptive", "released"]
        for n_obs in sorted({row["n_obs"] for row in effects_psnr})
        for region in ["full_frame", "flame_unsupported", "oral"]
    }
    plot_scaling(aggregate_psnr, args.output)
    plot_advantage_heatmaps(effects_psnr, args.output)
    plot_pareto(identity_rows, geometry_rows, args.output)
    report = {
        "kind": "identity_scaling_analysis",
        "run_count": len(runs),
        "matrix_complete": not missing,
        "n_ids": sorted({run["n_id"] for run in runs}),
        "methods": METHODS,
        "seeds": sorted({run["seed"] for run in runs}),
        "test_identity_count": len({row["subject"] for row in identity_rows}),
        "identity_level_metric_rows": len(identity_rows),
        "bootstrap": {
            "unit": "held-out identity; target cameras averaged within identity before resampling",
            "training_seed_handling": (
                "Paired method effects are averaged across training seeds within identity before "
                "identity bootstrap; per-seed identity-clustered effects are also reported."
            ),
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "crossovers": crossovers,
        "paired_psnr_effects": effects_psnr,
        "per_seed_paired_psnr_effects": per_seed_effects_psnr,
        "scaling_interactions_psnr": interactions_psnr,
        "paired_lpips_effects": effects_lpips,
        "scaling_interactions_lpips": interactions_lpips,
        "scaffold_quality_sensitivity": {
            "cache_audit": str(args.cache_audit) if args.cache_audit is not None else None,
            "flagged_subjects": sorted(flagged_subjects),
            "flagged_test_identity_count": len(
                flagged_subjects & {row["subject"] for row in identity_rows}
            ),
            "paired_psnr_effects": sensitivity_effects_psnr,
            "scaling_interactions_psnr": sensitivity_interactions_psnr,
            "policy": "Primary analysis retains all identities; this is a labeled sensitivity analysis only.",
        },
        "warning": (
            "Inference is over the 20 fixed official SVFR identities for one static FREE frame each. "
            "Camera views are not treated as independent samples."
        ),
    }
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "run_count": len(runs),
                "test_identity_count": report["test_identity_count"],
                "paired_effect_count": len(effects_psnr),
                "interaction_count": len(interactions_psnr),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
