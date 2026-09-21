# Real-tongue geometry release pilot

Case: subject `017`, `EXP-6-tongue-1`, frame `182`, 16 calibrated views, equal 4,000-step oracle fits.

## Result

- Unsupported-region PSNR improves from **11.45 dB** (hard) to **38.35 dB** (free), a **+26.91 dB** gap.
- The curve has a broad knee: 50 mm reaches 28.10 dB, 75 mm 33.22 dB, and 100 mm 35.23 dB.
- A 200 mm release is close to free on this frame: unsupported gap **+0.23 dB**; tongue-core gap **+0.34 dB**.
- The exploratory tongue-core contrast is **+10.61 dB** (free minus hard).

## Interpretation

This is direct evidence that tight FLAME support is a static representation bottleneck for a real protruding tongue. It also shows that the bottleneck is not binary: performance changes continuously with the allowed release radius, with most recovery occurring between 25 and 100 mm.

## Scope and caveats

- All 16 views are optimized and evaluated; this is an oracle ceiling test, not novel-view generalization.
- Geometry is a 16-silhouette visual-hull proxy, not scan ground truth; mouth concavities are not identifiable.
- The FLAME-distance supported/unsupported partition is the primary region definition.
- `tongue-core` is an exploratory, visually audited target-image ROI from a mouth-aligned geometric candidate plus color-guided GrabCut. It is not a substitute for manually verified semantic masks.
- This single subject/frame establishes a mechanism witness, not a population claim and not the animation-binding result.
