# Reproducibility ledger

Frozen summary on 2026-07-16. This ledger covers the public cross-identity NeRSemble study; earlier static-oracle and dynamic `C/A/P` experiments have their own artifact directories listed in `docs/07_current_findings.md`.

## Data contract

- Cache audit: `artifacts/analysis/nersemble_scale_cache/`
- Usable cached identities: 415
- Nested non-benchmark train identities: `8/32/128/384`
- Validation identities: 7
- Official SVFR test identities: 20, absent from every training level
- Common calibrated cameras: 15
- Observation prefixes: `N_obs={1,4,8}`; every target camera is excluded from its input prefix
- Static frame policy: maximum oral aperture among the fixed FREE candidates
- FLAME scaffold: multiview landmark fit, intentionally oracle-favorable to hard FLAME
- Leakage control: cache `initial_colors` is never read; target RGB is excluded from encoder inputs and slot sampling

Manifest: `artifacts/audits/nersemble_identity_scaling.json`. The 415-cache combined listing hash recorded during the audit is `7f13cf...`; see the audit artifact for the full digest and flagged-subject policy.

## Training matrix

| family | runs | steps | trainable parameters | Gaussian slots | head-ROI reevaluations | measured A800 GPU-hours |
|---|---:|---:|---:|---:|---:|---:|
| `identity_scaling20k_v1` | 36 | 20k | 651,816 | 5,023 | 36 | 1.81 |
| `identity_scaling20k_scale3_v1` | 18 | 20k | 651,816 | 5,023 | 18 | 0.91 |
| `identity_scaling20k_h64_v1` | 18 | 20k | 615,464 | 5,023 | 18 | 0.91 |
| `identity_scaling20k_h512_v1` | 18 | 20k | 1,213,992 | 5,023 | 18 | 0.96 |
| `identity_scaling20k_adaptive_r30_v1` | 6 | 20k | 651,816 | 5,023 | 6 | 0.33 |
| `identity_scaling20k_adaptive_r75_v1` | 6 | 20k | 651,816 | 5,023 | 6 | 0.34 |
| `identity_scaling60k_v1` | 9 | 60k | 651,816 | 5,023 | 9 | 1.38 |
| `identity_scaling20k_head_target_v1` | 18 | 20k | 651,816 | 5,023 | 18 | 1.01 |
| `identity_scaling20k_density2_v1` | 18 | 20k | 812,552 | 10,046 | 18 | 0.97 |
| **total** | **147** | — | — | — | **147** | **8.63** |

GPU-hours sum the `runtime_seconds` recorded by each single-GPU training run. They exclude cache construction, LPIPS/head-ROI re-evaluation, early smoke tests, static oracle experiments, and dynamic `C/A/P` experiments.

Every ordinary full run contains 4,480 test-region rows and 60 geometry rows. Every frozen head-ROI evaluation contains 2,560 rows. All 147 head evaluations were checked against the SHA-256 of their source `metrics.json` and `checkpoint.pt`; no non-finite result was found.

## Pre-registration and disclosure

- Primary protocol: `artifacts/audits/identity_scaling_preregistered_protocol.json`
- Covariance/capacity/convergence/dose follow-ups: `artifacts/audits/identity_scaling_followup_controls.json`
- Post-primary target-region control: `artifacts/audits/identity_scaling_target_region_preregister.json`
- Post-primary 2x-density control: `artifacts/audits/identity_scaling_density2_preregister.json`

The broad FLAME-unsupported mask is the pre-registered region. The head-versus-torso decomposition was added after qualitative audit and is always labeled post-audit. Target-region and density experiments were designed after the primary result, frozen before smoke/training, and labeled post-primary. The original result files were not overwritten.

## Statistical contract

- Target cameras are averaged within held-out identity.
- Paired training-seed effects are averaged within identity.
- The 20 held-out identities are resampled 10,000 times.
- Cameras and `identity × seed` rows are not treated as independent samples.
- Positive effect means comparison is better: comparison minus reference for PSNR/SSIM and reference minus comparison for MAE/LPIPS.
- Crossover is only reported as observed, left-censored, or right-censored over measured identity levels.
- Head-ROI spatial LPIPS on very narrow unsupported masks is interpreted cautiously because the AlexNet receptive field crosses the mask boundary.

## Analysis artifacts

| analysis | directory | central outputs |
|---|---|---|
| primary identity scaling | `artifacts/analysis/identity_scaling20k_v1/` | absolute metrics, paired effects, seed effects, interactions, scaffold sensitivity |
| post-audit head decomposition | `artifacts/analysis/identity_scaling20k_v1_head_roi/` | head supported/unsupported and torso curves |
| nuisance and release-dose controls | `artifacts/analysis/identity_scaling_followups_v1/` | covariance/capacity/convergence gap contrasts, dose paired effects and chord residuals |
| target and density controls | `artifacts/analysis/identity_scaling_mechanism_controls_v1/` | method absolute deltas, gap deltas, geometry deltas, per-seed effects |

## Current code fingerprints

| file | SHA-256 |
|---|---|
| `scripts/run_identity_scaling.py` | `63af95d03eb769cdb85cb64afd5483f84efdb53c73f715fad63203fed2f48f7d` |
| `scripts/reevaluate_identity_scaling_head_roi.py` | `741ce0218089492d836ca6ed6faabf10f3c806130567bdabe1d3769c9ce9e071` |
| `scripts/analyze_identity_scaling.py` | `8fee8504b06090fbca32f5e9f0a25bd1c0fc0b80a53686241bbfbae65886751c` |
| `scripts/analyze_identity_scaling_followups.py` | `a41ab0cc0ada31d62e707c92250018d104c61adfbff79e46688216152f24bb7d` |
| `scripts/analyze_identity_scaling_mechanism_controls.py` | `f4576c0138e731eecbe472f19cfc027ee31ceb91763634c07f4e4cea1acc9e5c` |
| `tests/test_research_tools.py` | `8d048ff7ffb491fb23a10be0a0523abdf5fbb46037dd64e7e456ffc07d889cb8` |

The target/density pre-registration files contain the training and reevaluation fingerprints frozen before their smoke tests. Later changes to analysis-only plotting do not change the trained checkpoints.

## Verification

```bash
scripts/run_py310.sh -m pytest -q tests/test_research_tools.py
```

Expected result at ledger freeze: `14 passed`.
