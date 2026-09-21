# `G_mu × G_Sigma × G_density` factorial

## Design

- Five landmark-anchored NeRSemble NVS cases.
- `mean={hard,free} × axis-cap={3,30} mm × density={0.25×,1×,4×}`.
- Within every density, hard/free have identical Gaussian counts, appearance parameters, optimizer, 4,000 steps, and all calibrated views.
- Low-density levels are nested FPS subsets; high density retains all FLAME vertices and adds deterministic area-sampled points on the same FLAME surface.

## Paired population effects

- Unsupported PSNR density gain (4× minus 0.25×), hard/3 mm: **+0.52 dB** [+0.15, +1.02].
- Unsupported PSNR density gain, hard/30 mm: **+3.82 dB** [+3.10, +4.38].
- Unsupported PSNR density gain, free/3 mm: **+18.98 dB** [+17.13, +20.83].
- Within hard means, density×extent interaction `(4×−0.25× at 30 mm) − (4×−0.25× at 3 mm)`: **+3.30 dB** [+2.07, +4.20].
- Unsupported three-way interaction `(mean×extent at 4×) − (mean×extent at 0.25×)`: **+15.18 dB** [+13.36, +17.05].
- Hard unsupported center-error change, 4× minus 0.25×: **-0.42 mm** [-0.55, -0.28].
- At 4×, absolute unsupported center error remains **35.84 mm** [31.17, 39.97] for hard versus **5.20 mm** [4.84, 5.57] for free.

## Interpretation rule

Density is a third geometry-capacity channel. A large hard density gain with wide but not tight covariance is evidence of a cardinality-amplified covariance bypass, not evidence that the FLAME support surface became correct. A residual hard center-error floor at 4× is the actual support ceiling. Intervals resample five subjects and remain descriptive.
