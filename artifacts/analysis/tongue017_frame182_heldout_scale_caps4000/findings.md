# Gaussian extent as a hidden geometry-release channel

Case: subject `017`, `EXP-6-tongue-1`, frame `182`, 16 calibrated views; all conditions use 5,023 Gaussians and 4,000 optimization steps.

## Factorial result

- With a 3 mm axis cap, free means beat hard means in the oral-protrusion ROI by **+4.63 dB**.
- With the original 30 mm cap, that same gap is only **+2.26 dB**.
- Raising only the scale cap from 3 to 30 mm changes hard-mean oral PSNR by **+4.31 dB**, versus **+1.94 dB** for free means. The difference in extent gains is **+2.37 dB**.
- Hard-mean unsupported center error remains 44.97–44.97 mm across scale caps: covariance can improve images without correcting center geometry.
- In the original hard/30 condition, the 90th percentile of the maximum axis within 45 mm of the fitted lips is **13.28 mm**.

## Finding

`FLAME geometry bias` is not one knob in Gaussian avatars. Mean anchoring and covariance extent are separate geometry channels. A model with hard FLAME means can bypass part of the apparent representation ceiling by stretching splats, producing plausible RGB while retaining wrong center geometry. Any FLAME-bias scaling law that varies only query/mean locations is confounded unless Gaussian extent is controlled and geometry-sensitive metrics are reported.


## Held-out cameras

- At a 3 mm extent cap, free-minus-hard oral PSNR is **+5.82 dB** on optimized cameras and **+1.04 dB** on held-out cameras.
- At 30 mm, the corresponding gaps are **+2.53 dB** and **+1.44 dB**.
- At 30 mm extent, 50 mm mean release is **+0.42 dB** versus free on held-out oral pixels, but **-1.80 dB** on the broader unsupported region. The region-specific optimum already differs in this case.


## Scope

This is a single-frame mechanism test. Twelve cameras are optimized and four angularly interleaved cameras are held out. It does not establish a population scaling law.
