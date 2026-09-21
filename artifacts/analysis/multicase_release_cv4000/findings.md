# Five-subject cross-validation of bounded release

## Design

- Five NeRSemble NVS cases; all 13 cameras per subject are held out exactly once across three angularly interleaved folds.
- Externally selected conditions from the tongue pilot: hard, 75 mm bounded mean release, and free means; 30 mm Gaussian-axis cap and 4,000 steps throughout.

## Result

- Supported region, 75 mm minus hard: **+7.10 dB**, subject bootstrap 95% CI [+4.69, +9.84].
- Supported region, 75 mm minus free: **+0.13 dB** [+0.01, +0.24].
- Unsupported region, 75 mm minus free: **-0.45 dB** [-0.89, -0.00].
- Unsupported region, free minus hard: **+11.69 dB** [+10.06, +14.20].
- Per-subject optimal-condition counts: supported `{'hard': 0, 'r075': 4, 'free': 1}`; unsupported `{'hard': 0, 'r075': 1, 'free': 4}`.

## Interpretation

This tests whether the region-dependent bounded-release witness transfers beyond the selecting tongue case. Positive bounded-vs-free supported effects would support regularization; negative unsupported effects would preserve the representation-ceiling side. Regions here are geometric distance bands, not manual semantic masks, so the result cannot replace oral/hair-specific replication.

## Scope

The subject bootstrap is descriptive with n=5. These are per-instance optimizations with held-out cameras, not a cross-identity trainable scaling law.
