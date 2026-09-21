# Gaussian extent as a hidden geometry-release channel

Case: subject `017`, `EXP-6-tongue-1`, frame `182`, 16 calibrated views; all conditions use 5,023 Gaussians and 4,000 optimization steps.

## Factorial result

- With a 3 mm axis cap, free means beat hard means in the oral-protrusion ROI by **+4.67 dB**.
- With the original 30 mm cap, that same gap is only **+1.15 dB**.
- Raising only the scale cap from 3 to 30 mm changes hard-mean oral PSNR by **+5.46 dB**, versus **+1.93 dB** for free means. The difference in extent gains is **+3.53 dB**.
- Hard-mean unsupported center error remains 44.97–44.97 mm across scale caps: covariance can improve images without correcting center geometry.
- In the original hard/30 condition, the 90th percentile of the maximum axis within 45 mm of the fitted lips is **14.41 mm**.

## Finding

`FLAME geometry bias` is not one knob in Gaussian avatars. Mean anchoring and covariance extent are separate geometry channels. A model with hard FLAME means can bypass part of the apparent representation ceiling by stretching splats, producing plausible RGB while retaining wrong center geometry. Any FLAME-bias scaling law that varies only query/mean locations is confounded unless Gaussian extent is controlled and geometry-sensitive metrics are reported.

## Scope

This is a single-frame, all-view oracle mechanism test. It establishes the confound and its direction, not a population scaling law or held-out-view result.
