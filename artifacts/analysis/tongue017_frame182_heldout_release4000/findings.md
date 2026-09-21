# Real-tongue geometry release pilot

Case: subject `017`, `EXP-6-tongue-1`, frame `182`, 16 calibrated views, equal 4,000-step oracle fits.

## Result

- Unsupported-region PSNR improves from **11.74 dB** (hard) to **35.47 dB** (free), a **+23.73 dB** gap.
- The curve has a broad knee: 50 mm reaches 28.54 dB, 75 mm 32.71 dB, and 100 mm 34.07 dB.
- A 200 mm release is close to free on this frame: unsupported gap **+0.45 dB**; oral-protrusion gap **+0.03 dB**.
- The exploratory oral-protrusion contrast is **+2.46 dB** (free minus hard).

## Interpretation

The FLAME-distance `unsupported` region gives a strong static support bottleneck, and the recovery is continuous rather than binary. However, the directly audited oral-protrusion ROI improves by only **+2.46 dB**. Therefore this run does **not** establish that hard FLAME means cannot reproduce tongue appearance. A hard-mean 3DGS can partly bypass its center constraint through learnable Gaussian extent/covariance. Scale-cap ablations are required before attributing the RGB curve to FLAME center support alone.

On held-out cameras, the best oral condition is **r010**, while the best broader unsupported condition is **free**. This is a single-case witness that the optimal prior strength can be region-dependent.

## Scope and caveats

- Twelve views are optimized and four angularly interleaved views are held out.
- Geometry is a 16-silhouette visual-hull proxy, not scan ground truth; mouth concavities are not identifiable.
- The FLAME-distance supported/unsupported partition is the primary region definition.
- `oral-protrusion` is an exploratory, visually audited polygon from the two mouth corners and a tongue-attracted chin landmark. It is not a substitute for manually verified semantic masks.
- This single subject/frame establishes a mechanism witness, not a population claim and not the animation-binding result.
