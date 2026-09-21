# Real-tongue geometry release pilot

Case: subject `017`, `EXP-6-tongue-1`, frame `182`, 16 calibrated views, equal 4,000-step oracle fits.

## Result

- Unsupported-region PSNR improves from **11.74 dB** (hard) to **36.25 dB** (free), a **+24.51 dB** gap.
- The curve has a broad knee: 50 mm reaches 28.90 dB, 75 mm 33.24 dB, and 100 mm 35.08 dB.
- A 200 mm release is close to free on this frame: unsupported gap **+0.36 dB**; oral-protrusion gap **-0.36 dB**.
- The exploratory oral-protrusion contrast is **+1.15 dB** (free minus hard).

## Interpretation

This is direct evidence that tight FLAME support is a static representation bottleneck for a real protruding tongue. It also shows that the bottleneck is not binary: performance changes continuously with the allowed release radius, with most recovery occurring between 25 and 100 mm.

## Scope and caveats

- All 16 views are optimized and evaluated; this is an oracle ceiling test, not novel-view generalization.
- Geometry is a 16-silhouette visual-hull proxy, not scan ground truth; mouth concavities are not identifiable.
- The FLAME-distance supported/unsupported partition is the primary region definition.
- `oral-protrusion` is an exploratory, visually audited polygon from the two mouth corners and a tongue-attracted chin landmark. It is not a substitute for manually verified semantic masks.
- This single subject/frame establishes a mechanism witness, not a population claim and not the animation-binding result.
