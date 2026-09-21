# Geometry oracle pilot: subject 388 / GLASSES / frame 0

This is a single-case, single-seed oracle diagnostic, not a population-level result.

- Common budget: 5023 Gaussians, 600 steps, 128 px image height, 13 calibrated views.
- Hard FLAME supported/unsupported PSNR: 27.07 / 12.19 dB.
- Free geometry supported/unsupported PSNR: 33.11 / 25.04 dB.
- Free-minus-hard gain: +6.04 dB supported, +12.84 dB unsupported, +11.64 dB whole head.
- Unsupported GT-to-mean geometry error: 40.88 -> 10.29 mm.
- Supported GT-to-mean geometry error: 3.91 -> 4.99 mm.

Interpretation: this case provides a positive signal for a geometry representation ceiling in hair/glasses/beard-like regions. It also shows a resource trade-off: free Gaussians improve unsupported coverage while slightly worsening supported geometric coverage. The finding must be repeated across seeds, subjects, regions, resolutions, Gaussian counts, and optimization budgets before it can support H2.
