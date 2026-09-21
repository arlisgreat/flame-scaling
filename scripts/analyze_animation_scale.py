#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flame_oracle.flame import load_flame_linear_assets


DEFAULT_ASSET_ROOT = Path(
    "/data1/workspace/airulan/3dv/external/LAM/model_zoo/human_parametric_models/flame_vhap"
)


def moving_block_interval(values: np.ndarray, block_length: int = 5, samples: int = 10000) -> list[float]:
    rng = np.random.default_rng(0)
    values = np.asarray(values, np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    starts = np.arange(max(1, len(values) - block_length + 1))
    draws = []
    blocks_per_draw = int(np.ceil(len(values) / block_length))
    for _ in range(samples):
        selected = rng.choice(starts, blocks_per_draw, replace=True)
        sample = np.concatenate([values[start : start + block_length] for start in selected])[: len(values)]
        draws.append(sample.mean())
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def first_crossover(scales: list[int], differences: list[float]) -> dict[str, object]:
    for left, right, y0, y1 in zip(scales[:-1], scales[1:], differences[:-1], differences[1:]):
        if y0 <= 0 < y1:
            fraction = -y0 / (y1 - y0)
            estimate = float(np.exp(np.log(left) + fraction * (np.log(right) - np.log(left))))
            return {"status": "observed", "bracket": [left, right], "log_interpolated_estimate": estimate}
    return {
        "status": "already_positive_at_smallest" if differences[0] > 0 else "right_censored",
        "boundary": scales[0] if differences[0] > 0 else scales[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-root", type=Path, default=Path("artifacts/runs/animation_scale_393_exp5"))
    parser.add_argument("--n24-root", type=Path, default=Path("artifacts/runs/animation_pilot_393_exp5_n24"))
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    methods = ("hard", "bounded", "adaptive", "free")
    method_labels = {"hard": "hard A", "bounded": "uniform residual", "adaptive": "adaptive release", "free": "free A"}
    paths = {
        8: {method: args.scale_root / f"n8_{method}" / "metrics.json" for method in methods},
        24: {
            "hard": args.n24_root / "hard" / "metrics.json",
            "bounded": args.n24_root / "bounded30" / "metrics.json",
            "adaptive": args.n24_root / "adaptive30" / "metrics.json",
            "free": args.n24_root / "free" / "metrics.json",
        },
        72: {method: args.scale_root / f"n72_{method}" / "metrics.json" for method in methods},
        155: {method: args.scale_root / f"nall_{method}" / "metrics.json" for method in methods},
    }
    reports = {
        scale: {method: json.loads(path.read_text()) for method, path in method_paths.items()}
        for scale, method_paths in paths.items()
    }
    scales = list(reports)
    regions = ("head", "face", "oral", "eyes")

    curves = {
        region: {
            method: [reports[scale][method]["evaluation"][region]["heldout"]["masked_psnr"] for scale in scales]
            for method in methods
        }
        for region in regions
    }
    train_curves = {
        region: {
            method: [reports[scale][method]["evaluation"][region]["train"]["masked_psnr"] for scale in scales]
            for method in methods
        }
        for region in regions
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {"hard": "#444444", "bounded": "#2a9d8f", "adaptive": "#e76f51", "free": "#457b9d"}
    markers = {"hard": "o", "bounded": "s", "adaptive": "D", "free": "^"}
    for axis, region in zip(axes.ravel(), regions):
        for method in methods:
            axis.plot(scales, curves[region][method], marker=markers[method], color=colors[method], label=method_labels[method])
        axis.set_xscale("log")
        axis.set_xticks(scales, labels=[str(scale) for scale in scales])
        axis.set_title(region)
        axis.set_xlabel("training frames")
        axis.set_ylabel("held-out temporal PSNR (dB)")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.output / "animation_scale_curves.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for method in methods:
        gaps = [train - held for train, held in zip(train_curves["oral"][method], curves["oral"][method])]
        axis.plot(scales, gaps, marker=markers[method], color=colors[method], label=method_labels[method])
    axis.set_xscale("log")
    axis.set_xticks(scales, labels=[str(scale) for scale in scales])
    axis.set_xlabel("training frames")
    axis.set_ylabel("oral train-held-out PSNR gap (dB)")
    axis.legend(fontsize=8)
    fig.savefig(args.output / "oral_generalization_gap.png", dpi=180)
    plt.close(fig)

    assets = load_flame_linear_assets(
        args.asset_root / "flame2023.pkl", args.asset_root / "FLAME_masks.pkl", 300, 100
    )
    gate_regions = {}
    for scale in scales:
        model = np.load(paths[scale]["adaptive"].parent / "model.npz")
        gates = model["release_gates"]
        gate_regions[str(scale)] = {
            region: {
                "mean": float(gates[assets["masks"][region]].mean()),
                "p90": float(np.percentile(gates[assets["masks"][region]], 90)),
            }
            for region in ("face", "lips", "eye_region", "scalp", "neck")
        }

    pairwise = {}
    for scale in scales:
        pairwise[str(scale)] = {}
        for region in regions:
            hard_frames = reports[scale]["hard"]["evaluation"][region]["per_frame"]
            hard_by_id = {row["frame"]: row for row in hard_frames if row["frame"] in set(reports[scale]["hard"]["heldout_indices"])}
            pairwise[str(scale)][region] = {}
            for method in ("bounded", "adaptive", "free"):
                method_frames = reports[scale][method]["evaluation"][region]["per_frame"]
                method_by_id = {row["frame"]: row for row in method_frames if row["frame"] in hard_by_id}
                frame_ids = sorted(set(hard_by_id) & set(method_by_id))
                differences = np.asarray(
                    [method_by_id[index]["masked_psnr"] - hard_by_id[index]["masked_psnr"] for index in frame_ids]
                )
                pairwise[str(scale)][region][f"{method}_minus_hard"] = {
                    "mean_db": float(differences.mean()),
                    "moving_block_95ci_db": moving_block_interval(differences),
                    "heldout_frames": len(frame_ids),
                    "block_length_in_sampled_frames": 5,
                }

    crossovers = {
        region: {
            method: first_crossover(
                scales,
                [value - hard for value, hard in zip(curves[region][method], curves[region]["hard"])],
            )
            for method in ("adaptive", "free")
        }
        for region in regions
    }
    scale_gain = {
        region: {
            method: float(curves[region][method][-1] - curves[region][method][0]) for method in methods
        }
        for region in regions
    }
    best_by_region_scale = {
        region: {
            str(scale): max(methods, key=lambda method: reports[scale][method]["evaluation"][region]["heldout"]["masked_psnr"])
            for scale in scales
        }
        for region in regions
    }

    result = {
        "kind": "animation_scale_analysis",
        "scales": scales,
        "heldout_psnr_curves": curves,
        "train_psnr_curves": train_curves,
        "scale_gain_smallest_to_largest_db": scale_gain,
        "paired_moving_block_intervals": pairwise,
        "crossovers_against_hard": crossovers,
        "best_method_by_region_and_scale": best_by_region_scale,
        "adaptive_release_gate_by_flame_region": gate_regions,
        "headline": {
            "oral_hard_gain_8_to_155_db": scale_gain["oral"]["hard"],
            "oral_adaptive_gain_8_to_155_db": scale_gain["oral"]["adaptive"],
            "oral_free_minus_hard_at_8_db": curves["oral"]["free"][0] - curves["oral"]["hard"][0],
            "oral_free_minus_hard_at_155_db": curves["oral"]["free"][-1] - curves["oral"]["hard"][-1],
            "oral_adaptive_minus_hard_at_155_db": curves["oral"]["adaptive"][-1] - curves["oral"]["hard"][-1],
            "head_free_minus_adaptive_at_155_db": curves["head"]["free"][-1] - curves["head"]["adaptive"][-1],
            "oral_adaptive_minus_free_at_155_db": curves["oral"]["adaptive"][-1] - curves["oral"]["free"][-1],
        },
        "interpretation": [
            "Hard A is sample efficient relative to fully free A at 8 frames.",
            "Hard A improves only weakly with scale, while learned residual/free mappings improve much more strongly.",
            "Adaptive/bounded release dominates hard A in oral and face regions across this tested ladder.",
            "At the largest scale free A is best for the whole head, while adaptive A remains best for oral detail: the optimal prior is region dependent.",
        ],
        "scope": "One identity and one mouth-motion sequence; confidence intervals account for temporal blocks, not identity variation.",
    }
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Animation-bias scale pilot",
        "",
        f"- Oral hard, 8->155 frames: {result['headline']['oral_hard_gain_8_to_155_db']:+.2f} dB.",
        f"- Oral adaptive, 8->155 frames: {result['headline']['oral_adaptive_gain_8_to_155_db']:+.2f} dB.",
        f"- Free-hard oral: {result['headline']['oral_free_minus_hard_at_8_db']:+.2f} dB at 8 frames, {result['headline']['oral_free_minus_hard_at_155_db']:+.2f} dB at 155 frames.",
        f"- Adaptive-hard oral at 155 frames: {result['headline']['oral_adaptive_minus_hard_at_155_db']:+.2f} dB.",
        f"- At 155 frames free beats adaptive on head by {result['headline']['head_free_minus_adaptive_at_155_db']:+.2f} dB, but adaptive beats free on oral by {result['headline']['oral_adaptive_minus_free_at_155_db']:+.2f} dB.",
        "",
        "Single-identity pilot only. The temporal moving-block intervals do not substitute for subject-level replication.",
    ]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), **result["headline"], "crossovers": crossovers}, indent=2))


if __name__ == "__main__":
    main()
