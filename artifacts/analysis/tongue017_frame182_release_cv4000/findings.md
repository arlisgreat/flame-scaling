# Four-fold angular cross-validation of mean release

## Result

- Every one of the 16 cameras is held out exactly once; each fold optimizes the other 12 cameras.
- The best mean release for oral held-out PSNR is **r075** at **27.11 dB**.
- It exceeds hard by **+1.54 dB**, camera bootstrap 95% CI [+0.26, +2.91], and free by **+0.63 dB** [-0.02, +1.36].
- The broader unsupported region instead prefers **free** at **27.62 dB**. Free exceeds the oral-optimal condition there by **+0.55 dB** [+0.08, +0.98].
- Non-dominated release doses across oral and unsupported held-out PSNR: `r075, r200, free`.

## Finding

The same representation and data budget has no globally optimal FLAME mean-release radius. Tight/bounded release regularizes the oral region, while unsupported outer-head content requires much freer support. This is an empirical region-conditioned bias-variance trade-off and a direct motivation for region-aware release; it is not yet evidence that a learned APR gate beats the oracle-selected fixed doses.

## Scope

This is one subject and one static frame. The camera-level bootstrap quantifies view variation only; subject/sequence replication and dynamic animation tests remain required.
