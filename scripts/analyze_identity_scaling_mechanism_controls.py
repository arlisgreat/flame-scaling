#!/usr/bin/env python3
"""Analyze post-primary target-region and population-density controls."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_identity_scaling_followups import (
    METRICS,
    LOWER_IS_BETTER,
    absolute_aggregates,
    bootstrap_mean,
    camera_averaged_rows,
    geometry_rows,
    paired_method_effects,
    seed_averaged_lookup,
    write_csv,
)


RUN_PATTERN = re.compile(
    r"nid(?P<n_id>\d+)_(?P<method>hard|adaptive|released)_seed(?P<seed>\d+)"
)
VARIANTS = {
    "primary_full_target_density1": "identity_scaling20k_v1",
    "head_target_density1": "identity_scaling20k_head_target_v1",
    "full_target_density2": "identity_scaling20k_density2_v1",
}
EXPECTED = {
    "primary_full_target_density1": 36,
    "head_target_density1": 18,
    "full_target_density2": 18,
}


def load_runs(root: Path, variant: str, head_roi: bool) -> list[dict[str, Any]]:
    filename = "head_roi_metrics.json" if head_roi else "metrics.json"
    output = []
    for path in sorted((root / VARIANTS[variant]).glob(f"*/{filename}")):
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
    if len(output) != EXPECTED[variant]:
        raise RuntimeError(
            f"{variant} {'head' if head_roi else 'ordinary'} runs={len(output)} "
            f"expected={EXPECTED[variant]}"
        )
    return output


def metric_advantage(comparison: float, reference: float, metric: str) -> float:
    return reference - comparison if metric in LOWER_IS_BETTER else comparison - reference


def paired_control_deltas(
    rows: list[dict[str, Any]], metric: str, rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    lookup = seed_averaged_lookup(rows, metric)
    reference = "primary_full_target_density1"
    output = []
    for control in ["head_target_density1", "full_target_density2"]:
        n_ids = sorted({key[1] for key in lookup if key[0] == control})
        for n_id in n_ids:
            for method in ["hard", "adaptive", "released"]:
                cells = sorted(
                    {(key[3], key[4]) for key in lookup if key[:3] == (control, n_id, method)}
                )
                for n_obs, region in cells:
                    control_prefix = (control, n_id, method, n_obs, region)
                    reference_prefix = (reference, n_id, method, n_obs, region)
                    subjects = sorted(
                        {key[5] for key in lookup if key[:5] == control_prefix}
                        & {key[5] for key in lookup if key[:5] == reference_prefix}
                    )
                    if not subjects:
                        continue
                    values = [
                        metric_advantage(
                            lookup[(*control_prefix, subject)],
                            lookup[(*reference_prefix, subject)],
                            metric,
                        )
                        for subject in subjects
                    ]
                    mean, low, high = bootstrap_mean(values, rng, samples)
                    output.append(
                        {
                            "control_variant": control,
                            "reference_variant": reference,
                            "n_id": n_id,
                            "method": method,
                            "n_obs": n_obs,
                            "region": region,
                            "metric": metric,
                            "control_advantage_mean": mean,
                            "ci_low": low,
                            "ci_high": high,
                            "identity_count": len(subjects),
                        }
                    )
    return output


def gap_control_contrasts(
    rows: list[dict[str, Any]], metric: str, rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    lookup = seed_averaged_lookup(rows, metric)
    reference = "primary_full_target_density1"
    output = []
    for control in ["head_target_density1", "full_target_density2"]:
        n_ids = sorted({key[1] for key in lookup if key[0] == control})
        for n_id in n_ids:
            cells = sorted({(key[3], key[4]) for key in lookup if key[0] == control})
            for n_obs, region in cells:
                for comparison in ["adaptive", "released"]:
                    prefixes = {
                        "control_hard": (control, n_id, "hard", n_obs, region),
                        "control_comparison": (control, n_id, comparison, n_obs, region),
                        "reference_hard": (reference, n_id, "hard", n_obs, region),
                        "reference_comparison": (reference, n_id, comparison, n_obs, region),
                    }
                    subject_sets = [
                        {key[5] for key in lookup if key[:5] == prefix}
                        for prefix in prefixes.values()
                    ]
                    subjects = sorted(set.intersection(*subject_sets))
                    if not subjects:
                        continue
                    values = []
                    for subject in subjects:
                        control_gap = metric_advantage(
                            lookup[(*prefixes["control_comparison"], subject)],
                            lookup[(*prefixes["control_hard"], subject)],
                            metric,
                        )
                        reference_gap = metric_advantage(
                            lookup[(*prefixes["reference_comparison"], subject)],
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
                            "comparison": comparison,
                            "n_obs": n_obs,
                            "region": region,
                            "metric": metric,
                            "change_in_comparison_advantage": mean,
                            "ci_low": low,
                            "ci_high": high,
                            "identity_count": len(subjects),
                        }
                    )
    return output


def geometry_control_deltas(
    rows: list[dict[str, Any]], rng: np.random.Generator, samples: int
) -> list[dict[str, Any]]:
    value_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.endswith("_offset_mean_mm") and isinstance(value, (int, float))
        }
    )
    output = []
    reference = "primary_full_target_density1"
    for value_key in value_keys:
        grouped: dict[tuple[str, int, str, int, str], list[float]] = defaultdict(list)
        for row in rows:
            if value_key in row:
                grouped[(row["variant"], row["n_id"], row["method"], row["n_obs"], row["subject"])].append(
                    float(row[value_key])
                )
        lookup = {key: float(np.mean(values)) for key, values in grouped.items()}
        for control in ["head_target_density1", "full_target_density2"]:
            n_ids = sorted({key[1] for key in lookup if key[0] == control})
            for n_id in n_ids:
                for method in ["hard", "adaptive", "released"]:
                    n_observations = sorted({key[3] for key in lookup if key[:3] == (control, n_id, method)})
                    for n_obs in n_observations:
                        control_prefix = (control, n_id, method, n_obs)
                        reference_prefix = (reference, n_id, method, n_obs)
                        subjects = sorted(
                            {key[4] for key in lookup if key[:4] == control_prefix}
                            & {key[4] for key in lookup if key[:4] == reference_prefix}
                        )
                        if not subjects:
                            continue
                        values = [
                            lookup[(*control_prefix, subject)]
                            - lookup[(*reference_prefix, subject)]
                            for subject in subjects
                        ]
                        mean, low, high = bootstrap_mean(values, rng, samples)
                        output.append(
                            {
                                "control_variant": control,
                                "reference_variant": reference,
                                "n_id": n_id,
                                "method": method,
                                "n_obs": n_obs,
                                "geometry_metric": value_key,
                                "control_minus_reference_mean": mean,
                                "ci_low": low,
                                "ci_high": high,
                                "identity_count": len(subjects),
                            }
                        )
    return output


def plot_gap_contrasts(rows: list[dict[str, Any]], output: Path) -> None:
    regions = ["head_foreground", "head_supported", "head_unsupported", "torso_outside_head_roi"]
    controls = ["head_target_density1", "full_target_density2"]
    labels = ["Head-only target", "2x density"]
    fig, axes = plt.subplots(2, len(regions), figsize=(14, 7), sharey="row")
    for metric_index, metric in enumerate(["masked_psnr", "lpips_alex_spatial"]):
        for region_index, region in enumerate(regions):
            ax = axes[metric_index, region_index]
            cells = [
                row for row in rows
                if row["metric"] == metric and row["n_id"] == 384 and row["n_obs"] == 4
                and row["region"] == region and row["comparison"] == "released"
            ]
            lookup = {row["control_variant"]: row for row in cells}
            y = np.arange(len(controls))
            means = np.asarray([lookup[name]["change_in_comparison_advantage"] for name in controls])
            lows = np.asarray([lookup[name]["ci_low"] for name in controls])
            highs = np.asarray([lookup[name]["ci_high"] for name in controls])
            ax.errorbar(means, y, xerr=[means - lows, highs - means], fmt="o", capsize=3)
            ax.axvline(0, color="black", linewidth=1, alpha=0.5)
            ax.set_yticks(y, labels)
            ax.set_title(region.replace("_", " "))
            ax.set_xlabel("Change in released-hard advantage")
            ax.grid(axis="x", alpha=0.2)
    axes[0, 0].set_ylabel("PSNR control")
    axes[1, 0].set_ylabel("LPIPS control")
    fig.suptitle("Target-region and density effects on the released-prior gap ($N_{id}=384$, $N_{obs}=4$)")
    fig.tight_layout()
    fig.savefig(output / "mechanism_control_gap_contrasts.png", dpi=220, bbox_inches="tight")
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

    ordinary_runs = []
    head_runs = []
    for variant in VARIANTS:
        ordinary_runs.extend(load_runs(args.runs_root, variant, head_roi=False))
        head_runs.extend(load_runs(args.runs_root, variant, head_roi=True))
    ordinary_rows = camera_averaged_rows(ordinary_runs, "ordinary")
    head_rows = camera_averaged_rows(head_runs, "head_roi")
    rows = ordinary_rows + head_rows
    geometry = geometry_rows(ordinary_runs)
    write_csv(args.output / "identity_metrics.csv", rows)
    write_csv(args.output / "identity_geometry.csv", geometry)

    aggregates = []
    method_effects = []
    per_seed_method_effects = []
    control_deltas = []
    gap_contrasts = []
    for metric in METRICS:
        aggregates.extend(absolute_aggregates(rows, metric, rng, args.bootstrap_samples))
        method_effects.extend(paired_method_effects(rows, metric, rng, args.bootstrap_samples))
        for training_seed in [0, 1, 2]:
            seed_rows = [row for row in rows if row["seed"] == training_seed]
            for row in paired_method_effects(seed_rows, metric, rng, args.bootstrap_samples):
                row["training_seed"] = training_seed
                per_seed_method_effects.append(row)
        control_deltas.extend(paired_control_deltas(rows, metric, rng, args.bootstrap_samples))
        gap_contrasts.extend(gap_control_contrasts(rows, metric, rng, args.bootstrap_samples))
    geometry_deltas = geometry_control_deltas(geometry, rng, args.bootstrap_samples)
    write_csv(args.output / "aggregate_metrics.csv", aggregates)
    write_csv(args.output / "paired_method_effects.csv", method_effects)
    write_csv(args.output / "per_seed_paired_method_effects.csv", per_seed_method_effects)
    write_csv(args.output / "paired_control_deltas.csv", control_deltas)
    write_csv(args.output / "gap_control_contrasts.csv", gap_contrasts)
    write_csv(args.output / "geometry_control_deltas.csv", geometry_deltas)
    plot_gap_contrasts(gap_contrasts, args.output)
    report = {
        "kind": "identity_scaling_target_region_and_density_control_analysis",
        "ordinary_runs": len(ordinary_runs),
        "head_roi_runs": len(head_runs),
        "bootstrap": {
            "unit": "held-out identity after camera and within-identity seed averaging",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "paired_method_effects": method_effects,
        "per_seed_paired_method_effects": per_seed_method_effects,
        "paired_control_deltas": control_deltas,
        "gap_control_contrasts": gap_contrasts,
        "geometry_control_deltas": geometry_deltas,
    }
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "ordinary_runs": len(ordinary_runs),
        "head_roi_runs": len(head_runs),
        "control_deltas": len(control_deltas),
        "gap_contrasts": len(gap_contrasts),
    }, indent=2))


if __name__ == "__main__":
    main()
