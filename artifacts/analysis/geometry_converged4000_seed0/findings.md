# Multi-case geometry oracle findings

## Evidence scope

- Five NeRSemble NVS subjects, 13 calibrated views per subject, one GT point cloud per case.
- 5,023 Gaussians and 50,230 total trainable parameters in every method.
- 4,000 optimization steps at 128 px image height; all methods use the same initial world-coordinate position LR.
- Hard has the same dormant position parameter tensor but zero active geometry DoF.

## Finding G1: tight FLAME geometry is a real static representation ceiling

- Free minus hard supported-region PSNR: +11.91 dB, bootstrap 95% CI [+10.22, +14.02].
- Free minus hard unsupported-region PSNR: +17.40 dB, bootstrap 95% CI [+15.28, +19.80].
- Free minus hard whole-head PSNR: +16.06 dB, bootstrap 95% CI [+14.31, +18.33].
- Hard minus free unsupported geometry error: +30.55 mm, bootstrap 95% CI [+24.87, +35.71].

## Finding G2: 200mm release is practically close to free geometry in this static oracle

- Free minus 200mm supported PSNR: +0.06 dB, bootstrap 95% CI [-0.09, +0.24].
- Free minus 200mm unsupported PSNR: +0.22 dB, bootstrap 95% CI [-0.01, +0.45].
- Free minus 200mm whole-head PSNR: +0.17 dB, bootstrap 95% CI [+0.11, +0.24].

The ±0.75 dB band in the figure is exploratory, not preregistered equivalence testing. The result says that FLAME geometry can be a bottleneck when kept tight, but FLAME query initialization is not itself a static ceiling once approximately 200mm release is available.

## What this does not establish

- The distance bands are FLAME-distance envelopes, not semantic ground-truth masks.
- All views are used for both oracle fitting and evaluation by design; this is not generalization evidence.
- Frame 0 of subject 445's tongue sequence does not visibly expose the tongue, so this run is not tongue evidence.
- This experiment does not release FLAME-semantic correspondence, LBS/blendshape animation, or tracker labels. Those are the next causal targets.
