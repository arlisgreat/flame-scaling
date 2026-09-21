# Multi-case mean-position × Gaussian-extent factorial

## Design

- Five NeRSemble NVS cases, each with a landmark-anchored, coefficient-bounded FLAME scaffold.
- Paired 2×2 intervention: Gaussian means hard vs free; maximum Gaussian-axis standard deviation 3 vs 30 mm.
- Every condition uses 5,023 Gaussians, identical trainable parameter counts, 4,000 steps, and all 13 calibrated views.

## Replicated finding

- Unsupported-region gain from relaxing extent, hard means: **+10.94 dB**, bootstrap 95% CI [+9.73, +12.17].
- The same extent gain with free means: **+9.14 dB**, bootstrap 95% CI [+7.85, +10.42].
- Mean-release gain at 3 mm extent: **+19.21 dB**, bootstrap 95% CI [+16.58, +21.86].
- Apparent mean-release gain at 30 mm extent: **+17.42 dB**, bootstrap 95% CI [+14.65, +20.64].
- Difference-in-differences (hard extent gain minus free extent gain): **+1.79 dB**, bootstrap 95% CI [+0.60, +2.98]; positive in **5/5** cases.

## Interpretation

The single-tongue-frame covariance bypass is replicated in every case. Gaussian extent is a second geometry channel whose capacity changes the measured benefit of releasing FLAME means. Therefore a one-dimensional hard-vs-free comparison does not identify a unique `FLAME geometry bias`; it measures a coupled mean-support/extent system.

Bootstrap intervals resample subjects and are descriptive with n=5. This remains an all-view oracle ceiling experiment, not held-out novel-view generalization.
