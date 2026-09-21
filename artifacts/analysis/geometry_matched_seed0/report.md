# Geometry oracle pilot: subject 388 / GLASSES / frame 0

This is a single-case, single-seed oracle diagnostic, not a population-level result.

- Common budget: 5023 Gaussians, 600 steps, 128 px image height, 13 calibrated views.
- Hard FLAME supported/unsupported PSNR: 27.07 / 12.19 dB.
- Free geometry supported/unsupported PSNR: 34.42 / 27.61 dB.
- Free-minus-hard gain: +7.36 dB supported, +15.42 dB unsupported, +13.83 dB whole head.
- Unsupported GT-to-mean geometry error: 40.88 -> 8.45 mm.
- Supported GT-to-mean geometry error: 3.91 -> 5.14 mm.
- 200mm-minus-free gap: -0.10 dB supported, -0.04 dB unsupported, -0.01 dB whole head.

Interpretation: tight geometry constraints create a clear ceiling in hair/glasses/beard-like regions, but a 200mm bounded offset closes the gap to free geometry in this static case. Therefore this run does **not** establish that FLAME query initialization itself is the bottleneck for methods that already allow approximately 200mm offsets. It redirects the dynamic hypothesis toward correspondence and animation binding. The result must still be repeated across seeds, subjects, regions, resolutions, Gaussian counts, and optimization budgets.
