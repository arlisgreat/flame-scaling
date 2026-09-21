#!/usr/bin/env python3
"""Export compact manuscript tables from frozen analysis CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def effect_cell(row: dict[str, str], mean_key: str) -> str:
    mean = float(row[mean_key])
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    return f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path("artifacts/analysis"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mechanism = args.analysis_root / "identity_scaling_mechanism_controls_v1"
    followups = args.analysis_root / "identity_scaling_followups_v1"

    effects = read_rows(mechanism / "paired_method_effects.csv")
    headline_regions = ["head_foreground", "head_supported", "head_unsupported", "torso_outside_head_roi"]
    headline_variants = [
        "primary_full_target_density1",
        "head_target_density1",
        "full_target_density2",
    ]
    phase_rows = [
        row for row in effects
        if row["comparison"] == "released" and row["n_obs"] == "4"
        and row["metric"] in {"masked_psnr", "lpips_alex_spatial"}
        and row["region"] in headline_regions and row["variant"] in headline_variants
    ]
    write_csv(args.output / "table_phase_effects.csv", phase_rows)
    phase_md = []
    for metric in ["masked_psnr", "lpips_alex_spatial"]:
        rows = []
        for variant in headline_variants:
            for n_id in [8, 384]:
                lookup = {
                    row["region"]: row for row in phase_rows
                    if row["metric"] == metric and row["variant"] == variant and int(row["n_id"]) == n_id
                }
                rows.append(
                    [variant, str(n_id)]
                    + [effect_cell(lookup[region], "advantage_over_hard_mean") for region in headline_regions]
                )
        phase_md.append(f"### {'PSNR' if metric == 'masked_psnr' else 'LPIPS utility'} released advantage over hard\n")
        phase_md.append(markdown_table(["variant", "N_id"] + headline_regions, rows))

    controls = read_rows(followups / "released_gap_control_contrasts.csv")
    control_variants = [
        "covariance_s3_h128_20k",
        "capacity_s5_h64_20k",
        "capacity_s5_h512_20k",
        "convergence_s5_h128_60k",
    ]
    control_rows = [
        row for row in controls
        if row["n_id"] == "384" and row["n_obs"] == "4"
        and row["metric"] in {"masked_psnr", "lpips_alex_spatial"}
        and row["region"] in headline_regions[:3] and row["control_variant"] in control_variants
    ]
    write_csv(args.output / "table_nuisance_control_contrasts.csv", control_rows)
    control_md = []
    for metric in ["masked_psnr", "lpips_alex_spatial"]:
        rows = []
        for variant in control_variants:
            lookup = {
                row["region"]: row for row in control_rows
                if row["metric"] == metric and row["control_variant"] == variant
            }
            rows.append(
                [variant]
                + [effect_cell(lookup[region], "change_in_released_advantage") for region in headline_regions[:3]]
            )
        control_md.append(f"### {'PSNR' if metric == 'masked_psnr' else 'LPIPS utility'} change in released-hard gap versus primary\n")
        control_md.append(markdown_table(["control"] + headline_regions[:3], rows))

    dose_effects = read_rows(followups / "adaptive_release_dose_paired_effects.csv")
    dose_frontier = read_rows(followups / "adaptive_release_dose_frontier_residuals.csv")
    conditions = ["adaptive_r30", "adaptive_r75", "adaptive_r150"]
    dose_effect_rows = [
        row for row in dose_effects
        if row["n_id"] == "384" and row["n_obs"] == "4" and row["reference"] == "hard"
        and row["metric"] in {"masked_psnr", "lpips_alex_spatial"}
        and row["region"] in headline_regions[:3] and row["condition"] in conditions
    ]
    dose_frontier_rows = [
        row for row in dose_frontier
        if row["n_id"] == "384" and row["n_obs"] == "4"
        and row["metric"] in {"masked_psnr", "lpips_alex_spatial"}
        and row["region"] in headline_regions[:3] and row["condition"] in conditions
    ]
    write_csv(args.output / "table_adaptive_dose_effects.csv", dose_effect_rows)
    write_csv(args.output / "table_adaptive_dose_frontier.csv", dose_frontier_rows)
    dose_md = []
    for condition in conditions:
        effects_lookup = {
            (row["metric"], row["region"]): row for row in dose_effect_rows if row["condition"] == condition
        }
        frontier_lookup = {
            (row["metric"], row["region"]): row for row in dose_frontier_rows if row["condition"] == condition
        }
        dose_md.append(f"#### {condition}\n")
        rows = []
        for region in headline_regions[:3]:
            psnr = effects_lookup[("masked_psnr", region)]
            lpips = effects_lookup[("lpips_alex_spatial", region)]
            psnr_frontier = frontier_lookup[("masked_psnr", region)]
            lpips_frontier = frontier_lookup[("lpips_alex_spatial", region)]
            rows.append(
                [
                    region,
                    effect_cell(psnr, "advantage_over_reference_mean"),
                    effect_cell(lpips, "advantage_over_reference_mean"),
                    effect_cell(psnr_frontier, "utility_above_hard_released_chord"),
                    effect_cell(lpips_frontier, "utility_above_hard_released_chord"),
                ]
            )
        dose_md.append(markdown_table(
            ["region", "PSNR vs hard", "LPIPS utility vs hard", "PSNR above chord", "LPIPS utility above chord"], rows
        ))

    geometry = read_rows(mechanism / "geometry_control_deltas.csv")
    geometry_keys = ["all_offset_mean_mm", "face_offset_mean_mm", "scalp_offset_mean_mm", "neck_offset_mean_mm"]
    geometry_rows = [
        row for row in geometry
        if row["n_id"] == "384" and row["n_obs"] == "4" and row["method"] in {"adaptive", "released"}
        and row["geometry_metric"] in geometry_keys
    ]
    write_csv(args.output / "table_geometry_control_deltas.csv", geometry_rows)

    report = [
        "# Manuscript tables exported from frozen analysis\n",
        "Positive values favor release/control for both PSNR and LPIPS utility. Intervals are held-out-identity bootstrap 95% CIs.\n",
        "## Regional phase effects\n",
        *phase_md,
        "\n## Nuisance controls\n",
        *control_md,
        "\n## Adaptive release dose at N_id=384\n",
        *dose_md,
    ]
    (args.output / "tables.md").write_text("\n\n".join(report) + "\n")
    print(f"wrote paper tables to {args.output}")


if __name__ == "__main__":
    main()
