#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_identity_scaling import (
    METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    aggregate_rows,
    identity_metric_rows,
    paired_effects,
    scaling_interactions,
    write_csv,
)


RUN_PATTERN = re.compile(r"nid(?P<n_id>\d+)_(?P<method>hard|adaptive|released)_seed(?P<seed>\d+)")


def load_runs(root: Path) -> list[dict[str, object]]:
    runs = []
    for path in sorted(root.glob("*/head_roi_metrics.json")):
        match = RUN_PATTERN.fullmatch(path.parent.name)
        if not match:
            continue
        runs.append(
            {
                "path": path,
                "n_id": int(match.group("n_id")),
                "method": match.group("method"),
                "seed": int(match.group("seed")),
                "payload": json.loads(path.read_text()),
            }
        )
    return runs


def plot_curves(aggregates: list[dict[str, object]], output: Path) -> None:
    regions = ["head_unsupported", "torso_outside_head_roi"]
    observations = sorted({int(row["n_obs"]) for row in aggregates})
    fig, axes = plt.subplots(2, len(observations), figsize=(4.2 * len(observations), 6.8), squeeze=False)
    for i, region in enumerate(regions):
        for j, n_obs in enumerate(observations):
            ax = axes[i, j]
            for method in METHODS:
                cells = sorted(
                    [
                        row
                        for row in aggregates
                        if row["region"] == region
                        and int(row["n_obs"]) == n_obs
                        and row["method"] == method
                    ],
                    key=lambda row: int(row["n_id"]),
                )
                x = np.asarray([int(row["n_id"]) for row in cells])
                y = np.asarray([float(row["mean"]) for row in cells])
                low = np.asarray([float(row["ci_low"]) for row in cells])
                high = np.asarray([float(row["ci_high"]) for row in cells])
                ax.plot(x, y, marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
                ax.fill_between(x, low, high, color=METHOD_COLORS[method], alpha=0.15)
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 32, 128, 384])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_title(f"{region}; $N_{{obs}}={n_obs}$")
            ax.set_xlabel("Training identities")
            ax.set_ylabel("PSNR (dB)")
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "head_vs_torso_scaling_psnr.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    expected = {
        (n_id, method, seed)
        for n_id in [8, 32, 128, 384]
        for method in METHODS
        for seed in [0, 1, 2]
    }
    observed = {(run["n_id"], run["method"], run["seed"]) for run in runs}
    if expected != observed:
        raise RuntimeError(f"head ROI matrix mismatch missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    identity_rows = identity_metric_rows(runs)
    rng = np.random.default_rng(args.seed)
    aggregate_psnr = aggregate_rows(identity_rows, "masked_psnr", rng, args.bootstrap_samples)
    effects_psnr = paired_effects(identity_rows, "masked_psnr", rng, args.bootstrap_samples)
    interactions_psnr = scaling_interactions(identity_rows, "masked_psnr", rng, args.bootstrap_samples)
    aggregate_lpips = aggregate_rows(identity_rows, "lpips_alex_spatial", rng, args.bootstrap_samples)
    effects_lpips = paired_effects(identity_rows, "lpips_alex_spatial", rng, args.bootstrap_samples)
    interactions_lpips = scaling_interactions(identity_rows, "lpips_alex_spatial", rng, args.bootstrap_samples)
    write_csv(args.output / "identity_metrics.csv", identity_rows)
    write_csv(args.output / "aggregate_psnr.csv", aggregate_psnr)
    write_csv(args.output / "paired_psnr_effects.csv", effects_psnr)
    write_csv(args.output / "scaling_interactions_psnr.csv", interactions_psnr)
    write_csv(args.output / "aggregate_lpips.csv", aggregate_lpips)
    write_csv(args.output / "paired_lpips_effects.csv", effects_lpips)
    write_csv(args.output / "scaling_interactions_lpips.csv", interactions_lpips)
    plot_curves(aggregate_psnr, args.output)
    report = {
        "kind": "post_audit_head_roi_identity_scaling_analysis",
        "run_count": len(runs),
        "test_identity_count": len({row["subject"] for row in identity_rows}),
        "paired_psnr_effects": effects_psnr,
        "scaling_interactions_psnr": interactions_psnr,
        "paired_lpips_effects": effects_lpips,
        "scaling_interactions_lpips": interactions_lpips,
        "status": "post-audit decomposition; original pre-registered broad analysis remains unchanged",
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "unit": "identity after camera and within-identity seed averaging",
            "seed": args.seed,
        },
    }
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(runs), "rows": len(identity_rows)}, indent=2))


if __name__ == "__main__":
    main()
