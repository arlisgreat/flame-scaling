#!/usr/bin/env python3
"""Joint analysis for the pre-registered identity-scaling follow-up controls.

The analysis unit is a held-out identity. Cameras are averaged first and the
three training seeds are averaged within identity before identity bootstrap.
This script intentionally keeps the primary 20k/5mm/h128 results as the frozen
reference and reports how the paired released-vs-hard gap changes under each
control.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


RUN_PATTERN = re.compile(
    r"nid(?P<n_id>\d+)_(?P<method>hard|adaptive|released)_seed(?P<seed>\d+)"
)
LOWER_IS_BETTER = {"masked_mae", "masked_mse", "alpha_mae", "lpips_alex_spatial"}
METRICS = ["masked_psnr", "masked_mae", "ssim", "lpips_alex_spatial"]

VARIANTS = {
    "primary_s5_h128_20k": "identity_scaling20k_v1",
    "covariance_s3_h128_20k": "identity_scaling20k_scale3_v1",
    "capacity_s5_h64_20k": "identity_scaling20k_h64_v1",
    "capacity_s5_h512_20k": "identity_scaling20k_h512_v1",
    "dose_r30_s5_h128_20k": "identity_scaling20k_adaptive_r30_v1",
    "dose_r75_s5_h128_20k": "identity_scaling20k_adaptive_r75_v1",
    "convergence_s5_h128_60k": "identity_scaling60k_v1",
}

EXPECTED_RUNS = {
    "primary_s5_h128_20k": 36,
    "covariance_s3_h128_20k": 18,
    "capacity_s5_h64_20k": 18,
    "capacity_s5_h512_20k": 18,
    "dose_r30_s5_h128_20k": 6,
    "dose_r75_s5_h128_20k": 6,
    "convergence_s5_h128_60k": 9,
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(
    values: Iterable[float], rng: np.random.Generator, samples: int
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def load_runs(runs_root: Path, variant: str, head_roi: bool) -> list[dict[str, Any]]:
    filename = "head_roi_metrics.json" if head_roi else "metrics.json"
    root = runs_root / VARIANTS[variant]
    output = []
    for path in sorted(root.glob(f"*/{filename}")):
        match = RUN_PATTERN.fullmatch(path.parent.name)
        if match is None:
            continue
        output.append(
            {
                "variant": variant,
                "n_id": int(match.group("n_id")),
                "method": match.group("method"),
                "seed": int(match.group("seed")),
                "payload": json.loads(path.read_text()),
                "path": str(path),
            }
        )
    return output


def validate_runs(runs: list[dict[str, Any]], variant: str, head_roi: bool) -> dict[str, Any]:
    expected = EXPECTED_RUNS[variant]
    if len(runs) != expected:
        kind = "head ROI" if head_roi else "primary-region"
        raise RuntimeError(f"{variant} {kind} run count {len(runs)} != {expected}")
    cells = [(run["n_id"], run["method"], run["seed"]) for run in runs]
    if len(cells) != len(set(cells)):
        raise RuntimeError(f"duplicate cells in {variant}")
    row_counts = {len(run["payload"]["test_rows"]) for run in runs}
    return {"run_count": len(runs), "test_row_counts": sorted(row_counts)}


def camera_averaged_rows(runs: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in run["payload"]["test_rows"]:
            groups[(str(row["subject"]), int(row["n_obs"]), str(row["region"]))].append(row)
        for (subject, n_obs, region), camera_rows in groups.items():
            record: dict[str, Any] = {
                "variant": run["variant"],
                "source": source,
                "n_id": run["n_id"],
                "method": run["method"],
                "seed": run["seed"],
                "subject": subject,
                "n_obs": n_obs,
                "region": region,
                "camera_count": len(camera_rows),
                "pixel_count_mean": float(np.mean([row["pixel_count"] for row in camera_rows])),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in camera_rows if metric in row]
                if values:
                    record[metric] = float(np.mean(values))
            output.append(record)
    return output


def geometry_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        for row in run["payload"].get("geometry_rows", []):
            output.append(
                {
                    "variant": run["variant"],
                    "n_id": run["n_id"],
                    "method": run["method"],
                    "seed": run["seed"],
                    "subject": str(row["subject"]),
                    "n_obs": int(row["n_obs"]),
                    **{f"all_{key}": value for key, value in row["all"].items()},
                    **{
                        f"{region}_{key}": value
                        for region, values in row["regions"].items()
                        for key, value in values.items()
                    },
                }
            )
    return output


def seed_averaged_lookup(
    rows: list[dict[str, Any]], metric: str
) -> dict[tuple[str, int, str, int, str, str], float]:
    grouped: dict[tuple[str, int, str, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if metric in row:
            grouped[
                (
                    row["variant"],
                    row["n_id"],
                    row["method"],
                    row["n_obs"],
                    row["region"],
                    row["subject"],
                )
            ].append(float(row[metric]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def absolute_aggregates(
    rows: list[dict[str, Any]], metric: str, rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    lookup = seed_averaged_lookup(rows, metric)
    grouped: dict[tuple[str, int, str, int, str], list[float]] = defaultdict(list)
    for (*cell, _subject), value in lookup.items():
        grouped[tuple(cell)].append(value)
    output = []
    for (variant, n_id, method, n_obs, region), values in sorted(grouped.items()):
        mean, low, high = bootstrap_mean(values, rng, samples)
        output.append(
            {
                "variant": variant,
                "n_id": n_id,
                "method": method,
                "n_obs": n_obs,
                "region": region,
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "identity_count": len(values),
            }
        )
    return output


def advantage(comparison: float, hard: float, metric: str) -> float:
    return hard - comparison if metric in LOWER_IS_BETTER else comparison - hard


def paired_method_effects(
    rows: list[dict[str, Any]], metric: str, rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    lookup = seed_averaged_lookup(rows, metric)
    cells = sorted({(key[0], key[1], key[3], key[4]) for key in lookup})
    output = []
    for variant, n_id, n_obs, region in cells:
        hard_prefix = (variant, n_id, "hard", n_obs, region)
        hard_subjects = {key[5] for key in lookup if key[:5] == hard_prefix}
        for comparison in ["adaptive", "released"]:
            comparison_prefix = (variant, n_id, comparison, n_obs, region)
            comparison_subjects = {key[5] for key in lookup if key[:5] == comparison_prefix}
            subjects = sorted(hard_subjects & comparison_subjects)
            if not subjects:
                continue
            effects = [
                advantage(
                    lookup[(*comparison_prefix, subject)],
                    lookup[(*hard_prefix, subject)],
                    metric,
                )
                for subject in subjects
            ]
            mean, low, high = bootstrap_mean(effects, rng, samples)
            output.append(
                {
                    "variant": variant,
                    "n_id": n_id,
                    "n_obs": n_obs,
                    "region": region,
                    "metric": metric,
                    "comparison": comparison,
                    "advantage_over_hard_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "identity_count": len(subjects),
                    "fraction_identities_advantage_positive": float(np.mean(np.asarray(effects) > 0)),
                }
            )
    return output


def released_gap_control_contrasts(
    rows: list[dict[str, Any]], metric: str, rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    """Difference of paired released-hard gaps between a control and primary."""
    lookup = seed_averaged_lookup(rows, metric)
    reference = "primary_s5_h128_20k"
    controls = [
        "covariance_s3_h128_20k",
        "capacity_s5_h64_20k",
        "capacity_s5_h512_20k",
        "convergence_s5_h128_60k",
    ]
    output = []
    for control in controls:
        control_n_ids = sorted({key[1] for key in lookup if key[0] == control})
        for n_id in control_n_ids:
            for n_obs, region in sorted({(key[3], key[4]) for key in lookup if key[0] == control}):
                prefixes = {
                    "control_hard": (control, n_id, "hard", n_obs, region),
                    "control_released": (control, n_id, "released", n_obs, region),
                    "reference_hard": (reference, n_id, "hard", n_obs, region),
                    "reference_released": (reference, n_id, "released", n_obs, region),
                }
                subject_sets = [
                    {key[5] for key in lookup if key[:5] == prefix}
                    for prefix in prefixes.values()
                ]
                if not subject_sets:
                    continue
                subjects = sorted(set.intersection(*subject_sets))
                if not subjects:
                    continue
                values = []
                for subject in subjects:
                    control_gap = advantage(
                        lookup[(*prefixes["control_released"], subject)],
                        lookup[(*prefixes["control_hard"], subject)],
                        metric,
                    )
                    reference_gap = advantage(
                        lookup[(*prefixes["reference_released"], subject)],
                        lookup[(*prefixes["reference_hard"], subject)],
                        metric,
                    )
                    values.append(control_gap - reference_gap)
                mean, low, high = bootstrap_mean(values, rng, samples)
                output.append(
                    {
                        "control_variant": control,
                        "reference_variant": reference,
                        "n_id": n_id,
                        "n_obs": n_obs,
                        "region": region,
                        "metric": metric,
                        "change_in_released_advantage": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "identity_count": len(subjects),
                    }
                )
    return output


def dose_rows(
    rows: list[dict[str, Any]], geometry: list[dict[str, Any]], metric: str,
    rng: np.random.Generator, samples: int,
) -> list[dict[str, Any]]:
    mapping = {
        ("primary_s5_h128_20k", "hard"): 0.0,
        ("dose_r30_s5_h128_20k", "adaptive"): 30.0,
        ("dose_r75_s5_h128_20k", "adaptive"): 75.0,
        ("primary_s5_h128_20k", "adaptive"): 150.0,
        ("primary_s5_h128_20k", "released"): 150.0,
    }
    metric_lookup = seed_averaged_lookup(rows, metric)
    geom_group: dict[tuple[str, int, str, int, str], list[float]] = defaultdict(list)
    for row in geometry:
        geom_group[(row["variant"], row["n_id"], row["method"], row["n_obs"], row["subject"])].append(
            float(row["all_offset_mean_mm"])
        )
    geom_lookup = {key: float(np.mean(values)) for key, values in geom_group.items()}
    output = []
    regions = sorted({key[4] for key in metric_lookup})
    for (variant, method), radius in mapping.items():
        for n_id in sorted({key[1] for key in metric_lookup if key[0] == variant and key[2] == method}):
            for n_obs in sorted({key[3] for key in metric_lookup if key[0] == variant and key[2] == method}):
                subjects_geom = [
                    key[4] for key in geom_lookup
                    if key[:4] == (variant, n_id, method, n_obs)
                ]
                mean_offset = float(np.mean([
                    geom_lookup[(variant, n_id, method, n_obs, subject)] for subject in subjects_geom
                ])) if subjects_geom else float("nan")
                for region in regions:
                    prefix = (variant, n_id, method, n_obs, region)
                    values = [value for key, value in metric_lookup.items() if key[:5] == prefix]
                    if not values:
                        continue
                    mean, low, high = bootstrap_mean(values, rng, samples)
                    output.append(
                        {
                            "variant": variant,
                            "method": method,
                            "release_radius_mm": radius,
                            "n_id": n_id,
                            "n_obs": n_obs,
                            "region": region,
                            "metric": metric,
                            "mean": mean,
                            "ci_low": low,
                            "ci_high": high,
                            "mean_offset_mm": mean_offset,
                            "identity_count": len(values),
                        }
                    )
    return output


DOSE_CONDITIONS = {
    "hard": ("primary_s5_h128_20k", "hard", 0.0),
    "adaptive_r30": ("dose_r30_s5_h128_20k", "adaptive", 30.0),
    "adaptive_r75": ("dose_r75_s5_h128_20k", "adaptive", 75.0),
    "adaptive_r150": ("primary_s5_h128_20k", "adaptive", 150.0),
    "released_r150": ("primary_s5_h128_20k", "released", 150.0),
}


def dose_paired_effects(
    rows: list[dict[str, Any]], metric: str, rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    lookup = seed_averaged_lookup(rows, metric)
    output = []
    for condition, (variant, method, radius) in DOSE_CONDITIONS.items():
        if condition == "hard":
            continue
        for reference_condition in ["hard", "released_r150"]:
            if condition == reference_condition:
                continue
            reference_variant, reference_method, _ = DOSE_CONDITIONS[reference_condition]
            n_ids = sorted({key[1] for key in lookup if key[0] == variant and key[2] == method})
            for n_id in n_ids:
                cells = sorted({(key[3], key[4]) for key in lookup if key[0] == variant and key[1] == n_id and key[2] == method})
                for n_obs, region in cells:
                    prefix = (variant, n_id, method, n_obs, region)
                    reference_prefix = (reference_variant, n_id, reference_method, n_obs, region)
                    subjects = sorted(
                        {key[5] for key in lookup if key[:5] == prefix}
                        & {key[5] for key in lookup if key[:5] == reference_prefix}
                    )
                    if not subjects:
                        continue
                    values = [
                        advantage(
                            lookup[(*prefix, subject)],
                            lookup[(*reference_prefix, subject)],
                            metric,
                        )
                        for subject in subjects
                    ]
                    mean, low, high = bootstrap_mean(values, rng, samples)
                    output.append(
                        {
                            "condition": condition,
                            "reference": reference_condition,
                            "release_radius_mm": radius,
                            "n_id": n_id,
                            "n_obs": n_obs,
                            "region": region,
                            "metric": metric,
                            "advantage_over_reference_mean": mean,
                            "ci_low": low,
                            "ci_high": high,
                            "identity_count": len(subjects),
                        }
                    )
    return output


def dose_frontier_residuals(
    rows: list[dict[str, Any]], geometry: list[dict[str, Any]], metric: str,
    rng: np.random.Generator, samples: int,
) -> list[dict[str, Any]]:
    """Utility above the per-identity hard-to-released linear displacement chord."""
    lookup = seed_averaged_lookup(rows, metric)
    geom_group: dict[tuple[str, int, str, int, str], list[float]] = defaultdict(list)
    for row in geometry:
        geom_group[(row["variant"], row["n_id"], row["method"], row["n_obs"], row["subject"])].append(
            float(row["all_offset_mean_mm"])
        )
    geom_lookup = {key: float(np.mean(values)) for key, values in geom_group.items()}
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    output = []
    hard_variant, hard_method, _ = DOSE_CONDITIONS["hard"]
    free_variant, free_method, _ = DOSE_CONDITIONS["released_r150"]
    for condition in ["adaptive_r30", "adaptive_r75", "adaptive_r150"]:
        variant, method, radius = DOSE_CONDITIONS[condition]
        n_ids = sorted({key[1] for key in lookup if key[0] == variant and key[2] == method})
        for n_id in n_ids:
            cells = sorted({(key[3], key[4]) for key in lookup if key[0] == variant and key[1] == n_id and key[2] == method})
            for n_obs, region in cells:
                prefixes = {
                    "hard": (hard_variant, n_id, hard_method, n_obs, region),
                    "adaptive": (variant, n_id, method, n_obs, region),
                    "released": (free_variant, n_id, free_method, n_obs, region),
                }
                metric_subjects = [
                    {key[5] for key in lookup if key[:5] == prefix}
                    for prefix in prefixes.values()
                ]
                geom_subjects = [
                    {key[4] for key in geom_lookup if key[:4] == prefix[:4]}
                    for prefix in prefixes.values()
                ]
                subjects = sorted(set.intersection(*(metric_subjects + geom_subjects)))
                if not subjects:
                    continue
                residuals = []
                fractions = []
                for subject in subjects:
                    offsets = {
                        name: geom_lookup[(prefix[0], prefix[1], prefix[2], prefix[3], subject)]
                        for name, prefix in prefixes.items()
                    }
                    denominator = offsets["released"] - offsets["hard"]
                    if denominator <= 1e-8:
                        continue
                    fraction = (offsets["adaptive"] - offsets["hard"]) / denominator
                    utilities = {
                        name: sign * lookup[(*prefix, subject)] for name, prefix in prefixes.items()
                    }
                    chord = utilities["hard"] + fraction * (utilities["released"] - utilities["hard"])
                    residuals.append(utilities["adaptive"] - chord)
                    fractions.append(fraction)
                if not residuals:
                    continue
                mean, low, high = bootstrap_mean(residuals, rng, samples)
                output.append(
                    {
                        "condition": condition,
                        "release_radius_mm": radius,
                        "n_id": n_id,
                        "n_obs": n_obs,
                        "region": region,
                        "metric": metric,
                        "utility_above_hard_released_chord": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "mean_displacement_fraction": float(np.mean(fractions)),
                        "identity_count": len(residuals),
                        "positive_means_better_quality_at_matched_linear_displacement": True,
                    }
                )
    return output


def plot_control_effects(rows: list[dict[str, Any]], output: Path) -> None:
    regions = ["head_foreground", "head_supported", "head_unsupported"]
    controls = [
        "covariance_s3_h128_20k",
        "capacity_s5_h64_20k",
        "capacity_s5_h512_20k",
        "convergence_s5_h128_60k",
    ]
    labels = ["sigma 3 vs 5 mm", "h64 vs h128", "h512 vs h128", "60k vs 20k"]
    fig, axes = plt.subplots(1, len(regions), figsize=(13.2, 4.2), sharey=True)
    for ax, region in zip(axes, regions):
        cells = [
            row for row in rows
            if row["metric"] == "masked_psnr" and row["n_obs"] == 4
            and row["n_id"] == 384 and row["region"] == region
        ]
        lookup = {row["control_variant"]: row for row in cells}
        y = np.arange(len(controls))
        means = np.asarray([lookup[name]["change_in_released_advantage"] for name in controls])
        lows = np.asarray([lookup[name]["ci_low"] for name in controls])
        highs = np.asarray([lookup[name]["ci_high"] for name in controls])
        ax.errorbar(means, y, xerr=[means - lows, highs - means], fmt="o", color="#2A9D8F", capsize=3)
        ax.axvline(0, color="black", linewidth=1, alpha=0.5)
        ax.set_yticks(y, labels)
        ax.set_title(region.replace("_", " "))
        ax.set_xlabel("Change in released-hard PSNR gap (dB)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Does the released-prior advantage survive nuisance controls? ($N_{id}=384$, $N_{obs}=4$)")
    fig.tight_layout()
    fig.savefig(output / "followup_control_contrasts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dose_pareto(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, n_id in zip(axes, [8, 384]):
        cells = [
            row for row in rows
            if row["metric"] == "masked_psnr" and row["n_id"] == n_id
            and row["n_obs"] == 4 and row["region"] == "head_foreground"
        ]
        endpoint_lookup = {row["method"]: row for row in cells if row["method"] in {"hard", "released"}}
        if {"hard", "released"} <= set(endpoint_lookup):
            hard = endpoint_lookup["hard"]
            released = endpoint_lookup["released"]
            ax.plot(
                [hard["mean_offset_mm"], released["mean_offset_mm"]],
                [hard["mean"], released["mean"]],
                linestyle="--", color="#666666", linewidth=1.2, alpha=0.75,
                label="hard-to-released chord",
            )
        for row in sorted(cells, key=lambda item: (item["mean_offset_mm"], item["method"])):
            marker = "s" if row["method"] == "hard" else ("^" if row["method"] == "released" else "o")
            ax.errorbar(
                row["mean_offset_mm"], row["mean"],
                yerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                fmt=marker, markersize=7, capsize=2,
            )
            label = "hard" if row["method"] == "hard" else (
                "released" if row["method"] == "released" else f"adaptive r={row['release_radius_mm']:.0f}"
            )
            offset = {
                "hard": (5, 4),
                "released": (-42, 10),
                "adaptive r=30": (5, 5),
                "adaptive r=75": (5, 5),
                "adaptive r=150": (6, -15),
            }[label]
            ax.annotate(
                label, (row["mean_offset_mm"], row["mean"]), xytext=offset,
                textcoords="offset points", fontsize=8,
            )
        ax.set_title(f"Training identities: {n_id}")
        ax.set_xlabel("Mean displacement from FLAME (mm)")
        ax.set_ylabel("Head-foreground PSNR (dB)")
        ax.grid(alpha=0.2)
    fig.suptitle("Adaptive release dose: quality-displacement frontier")
    fig.tight_layout()
    fig.savefig(output / "adaptive_release_dose_pareto.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ordinary_runs: list[dict[str, Any]] = []
    head_runs: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}
    for variant in VARIANTS:
        ordinary = load_runs(args.runs_root, variant, head_roi=False)
        head = load_runs(args.runs_root, variant, head_roi=True)
        integrity[variant] = {
            "ordinary": validate_runs(ordinary, variant, head_roi=False),
            "head_roi": validate_runs(head, variant, head_roi=True),
        }
        ordinary_runs.extend(ordinary)
        head_runs.extend(head)

    ordinary_rows = camera_averaged_rows(ordinary_runs, source="preregistered_regions")
    head_rows = camera_averaged_rows(head_runs, source="post_audit_head_roi")
    all_rows = ordinary_rows + head_rows
    geometry = geometry_rows(ordinary_runs)
    write_csv(args.output / "identity_metrics_all_controls.csv", all_rows)
    write_csv(args.output / "identity_geometry_all_controls.csv", geometry)

    aggregates: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    dose: list[dict[str, Any]] = []
    dose_effects: list[dict[str, Any]] = []
    dose_frontier: list[dict[str, Any]] = []
    for metric in METRICS:
        aggregates.extend(absolute_aggregates(all_rows, metric, rng, args.bootstrap_samples))
        effects.extend(paired_method_effects(all_rows, metric, rng, args.bootstrap_samples))
        contrasts.extend(released_gap_control_contrasts(all_rows, metric, rng, args.bootstrap_samples))
        dose.extend(dose_rows(all_rows, geometry, metric, rng, args.bootstrap_samples))
        dose_effects.extend(dose_paired_effects(all_rows, metric, rng, args.bootstrap_samples))
        dose_frontier.extend(dose_frontier_residuals(all_rows, geometry, metric, rng, args.bootstrap_samples))
    write_csv(args.output / "aggregate_metrics.csv", aggregates)
    write_csv(args.output / "paired_method_effects.csv", effects)
    write_csv(args.output / "released_gap_control_contrasts.csv", contrasts)
    write_csv(args.output / "adaptive_release_dose.csv", dose)
    write_csv(args.output / "adaptive_release_dose_paired_effects.csv", dose_effects)
    write_csv(args.output / "adaptive_release_dose_frontier_residuals.csv", dose_frontier)
    plot_control_effects(contrasts, args.output)
    plot_dose_pareto(dose, args.output)

    report = {
        "kind": "identity_scaling_followup_control_analysis",
        "integrity": integrity,
        "bootstrap": {
            "unit": "held-out identity after camera and within-identity seed averaging",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "ordinary_run_count": len(ordinary_runs),
        "head_roi_run_count": len(head_runs),
        "paired_method_effects": effects,
        "released_gap_control_contrasts": contrasts,
        "adaptive_release_dose": dose,
        "adaptive_release_dose_paired_effects": dose_effects,
        "adaptive_release_dose_frontier_residuals": dose_frontier,
        "interpretation_contract": (
            "Control contrasts are differences of paired released-vs-hard gaps relative to the "
            "frozen primary 20k/5mm/h128 experiment. Zero means the main gap is unchanged."
        ),
    }
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "ordinary_runs": len(ordinary_runs),
        "head_roi_runs": len(head_runs),
        "effects": len(effects),
        "contrasts": len(contrasts),
        "dose_rows": len(dose),
        "dose_effects": len(dose_effects),
        "dose_frontier_rows": len(dose_frontier),
    }, indent=2))


if __name__ == "__main__":
    main()
