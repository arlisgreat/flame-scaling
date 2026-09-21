#!/usr/bin/env python3
"""Integrity and quality audit for the static NeRSemble identity cache.

Quality thresholds flag sensitivity subsets only.  They never silently remove
an identity from the primary train/validation/test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--triangulation-threshold-px", type=float, default=3.0)
    parser.add_argument("--fit-mean-threshold-mm", type=float, default=8.0)
    parser.add_argument("--minimum-usable-views", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected = sorted(
        subject for subject, row in manifest["per_subject"].items() if row["usable"]
    )
    rows = []
    missing = []
    integrity_failures = []
    for subject in expected:
        report_path = args.cache_root / f"{subject}.json"
        cache_path = args.cache_root / f"{subject}.npz"
        if not report_path.exists() or not cache_path.exists():
            missing.append(subject)
            continue
        try:
            report = json.loads(report_path.read_text())
            with np.load(cache_path) as payload:
                required = [
                    "images",
                    "alpha",
                    "Ks",
                    "viewmats",
                    "vertices",
                    "faces",
                    "target_landmarks",
                    "predicted_landmarks",
                ]
                absent = [key for key in required if key not in payload]
                finite = all(
                    np.isfinite(payload[key]).all()
                    for key in required
                    if key in payload and payload[key].dtype.kind in "fc"
                )
                if absent or not finite:
                    integrity_failures.append(
                        {"subject": subject, "absent": absent, "all_numeric_finite": finite}
                    )
                rows.append(
                    {
                        "subject": subject,
                        "triangulation_median_px": float(
                            report["triangulation"]["median_landmark_median_reprojection_px"]
                        ),
                        "fit_mean_mm": float(report["fit"]["landmark_mean_mm"]),
                        "fit_median_mm": float(report["fit"]["landmark_median_mm"]),
                        "fit_max_mm": float(report["fit"]["landmark_max_mm"]),
                        "usable_views": int(report["landmark_detector_usable_views"]),
                        "oral_aperture_ratio": float(report["chosen_oral_aperture_ratio"]),
                        "frame": int(report["chosen_frame"]),
                        "camera_count": int(len(payload["camera_ids"])),
                        "gaussian_count": int(len(payload["vertices"])),
                    }
                )
        except Exception as error:
            integrity_failures.append({"subject": subject, "error": repr(error)})
    if missing:
        raise RuntimeError(f"cache is incomplete: {len(missing)} missing, first={missing[:10]}")
    if integrity_failures:
        raise RuntimeError(f"cache integrity failures: {integrity_failures[:5]}")

    flagged = [
        {
            **row,
            "reasons": [
                reason
                for condition, reason in [
                    (
                        row["triangulation_median_px"] > args.triangulation_threshold_px,
                        "triangulation_median_px",
                    ),
                    (row["fit_mean_mm"] > args.fit_mean_threshold_mm, "fit_mean_mm"),
                    (row["usable_views"] < args.minimum_usable_views, "usable_views"),
                ]
                if condition
            ],
        }
        for row in rows
        if row["triangulation_median_px"] > args.triangulation_threshold_px
        or row["fit_mean_mm"] > args.fit_mean_threshold_mm
        or row["usable_views"] < args.minimum_usable_views
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "nersemble_identity_scale_cache_audit",
        "expected_subject_count": len(expected),
        "complete_subject_count": len(rows),
        "missing_subjects": missing,
        "integrity_failure_count": len(integrity_failures),
        "thresholds_for_sensitivity_flag_only": {
            "triangulation_median_px": args.triangulation_threshold_px,
            "fit_mean_mm": args.fit_mean_threshold_mm,
            "minimum_usable_views": args.minimum_usable_views,
        },
        "statistics": {
            key: describe([row[key] for row in rows])
            for key in [
                "triangulation_median_px",
                "fit_mean_mm",
                "fit_median_mm",
                "fit_max_mm",
                "usable_views",
                "oral_aperture_ratio",
            ]
        },
        "flagged_subjects": flagged,
        "flagged_subject_count": len(flagged),
        "split_flag_counts": {
            split: len(set(subjects) & {row["subject"] for row in flagged})
            for split, subjects in {
                "train_384": manifest["nested_train_sets"]["384"],
                "validation": manifest["validation_subjects"],
                "svfr_test": manifest["usable_svfr_test"],
            }.items()
        },
        "primary_analysis_policy": (
            "Keep every pre-registered identity. Flagged thresholds define a disclosed sensitivity "
            "analysis only; no sample may be silently dropped after viewing model results."
        ),
        "scope_warning": (
            "Landmark fit residual is a scaffold quality diagnostic, not ground-truth surface error. "
            "Large residuals can be localized to jaw-outline landmarks while the head scaffold remains usable."
        ),
    }
    (args.output / "cache_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, label in zip(
        axes,
        ["triangulation_median_px", "fit_mean_mm", "fit_max_mm"],
        ["Triangulation median (px)", "FLAME landmark mean (mm)", "FLAME landmark max (mm)"],
    ):
        ax.hist([row[key] for row in rows], bins=30, color="#345995", alpha=0.85)
        ax.set_xlabel(label)
        ax.set_ylabel("Identities")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output / "cache_quality_distributions.png", dpi=220)
    plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete_subject_count": len(rows),
                "flagged_subject_count": len(flagged),
                "split_flag_counts": report["split_flag_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
