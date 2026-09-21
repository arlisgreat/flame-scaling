# Manuscript tables exported from frozen analysis


Positive values favor release/control for both PSNR and LPIPS utility. Intervals are held-out-identity bootstrap 95% CIs.


## Regional phase effects


### PSNR released advantage over hard


| variant | N_id | head_foreground | head_supported | head_unsupported | torso_outside_head_roi |
|---|---|---|---|---|---|
| primary_full_target_density1 | 8 | -0.09 [-0.63, +0.43] | -2.37 [-2.99, -1.85] | +1.51 [+0.86, +2.23] | +2.89 [+2.00, +3.75] |
| primary_full_target_density1 | 384 | +2.46 [+1.70, +3.24] | -0.89 [-1.28, -0.53] | +5.12 [+3.89, +6.39] | +7.09 [+5.68, +8.41] |
| head_target_density1 | 8 | +0.57 [+0.23, +0.94] | -0.85 [-1.46, -0.34] | +1.21 [+0.74, +1.75] | +0.13 [+0.01, +0.26] |
| head_target_density1 | 384 | +3.34 [+2.57, +4.16] | +0.57 [+0.36, +0.76] | +4.92 [+3.92, +5.97] | -0.19 [-0.38, +0.02] |
| full_target_density2 | 8 | -0.27 [-0.80, +0.24] | -2.31 [-2.92, -1.74] | +0.91 [+0.40, +1.45] | +3.22 [+2.39, +4.05] |
| full_target_density2 | 384 | +2.63 [+1.84, +3.41] | -0.48 [-0.75, -0.25] | +4.79 [+3.74, +5.92] | +7.12 [+6.01, +8.21] |

### LPIPS utility released advantage over hard


| variant | N_id | head_foreground | head_supported | head_unsupported | torso_outside_head_roi |
|---|---|---|---|---|---|
| primary_full_target_density1 | 8 | -0.05 [-0.06, -0.04] | -0.05 [-0.06, -0.04] | -0.05 [-0.07, -0.04] | -0.00 [-0.04, +0.03] |
| primary_full_target_density1 | 384 | -0.06 [-0.07, -0.05] | -0.06 [-0.07, -0.05] | -0.04 [-0.06, -0.03] | +0.05 [+0.02, +0.07] |
| head_target_density1 | 8 | +0.00 [-0.00, +0.01] | +0.00 [-0.00, +0.01] | +0.00 [-0.01, +0.01] | +0.00 [-0.00, +0.01] |
| head_target_density1 | 384 | -0.01 [-0.02, +0.00] | -0.01 [-0.02, +0.00] | +0.00 [-0.01, +0.01] | -0.00 [-0.02, +0.01] |
| full_target_density2 | 8 | -0.03 [-0.04, -0.02] | -0.03 [-0.04, -0.02] | -0.04 [-0.05, -0.02] | -0.01 [-0.05, +0.02] |
| full_target_density2 | 384 | -0.03 [-0.04, -0.02] | -0.03 [-0.04, -0.02] | -0.02 [-0.03, -0.01] | +0.03 [+0.00, +0.06] |


## Nuisance controls


### PSNR change in released-hard gap versus primary


| control | head_foreground | head_supported | head_unsupported |
|---|---|---|---|
| covariance_s3_h128_20k | +0.87 [+0.45, +1.25] | +2.21 [+1.74, +2.65] | +0.51 [+0.10, +0.95] |
| capacity_s5_h64_20k | -0.14 [-0.25, -0.03] | -0.08 [-0.19, +0.04] | -0.17 [-0.31, -0.04] |
| capacity_s5_h512_20k | +0.33 [+0.15, +0.53] | +0.21 [+0.06, +0.38] | +0.48 [+0.16, +0.85] |
| convergence_s5_h128_60k | +0.58 [+0.28, +0.90] | +0.47 [+0.11, +0.79] | +0.66 [+0.27, +1.10] |

### LPIPS utility change in released-hard gap versus primary


| control | head_foreground | head_supported | head_unsupported |
|---|---|---|---|
| covariance_s3_h128_20k | +0.09 [+0.07, +0.10] | +0.09 [+0.07, +0.11] | +0.06 [+0.04, +0.08] |
| capacity_s5_h64_20k | -0.01 [-0.01, -0.00] | -0.01 [-0.01, -0.00] | -0.01 [-0.01, -0.00] |
| capacity_s5_h512_20k | +0.02 [+0.01, +0.03] | +0.02 [+0.01, +0.03] | +0.02 [+0.01, +0.03] |
| convergence_s5_h128_60k | +0.04 [+0.03, +0.06] | +0.04 [+0.03, +0.06] | +0.05 [+0.03, +0.06] |


## Adaptive release dose at N_id=384


#### adaptive_r30


| region | PSNR vs hard | LPIPS utility vs hard | PSNR above chord | LPIPS utility above chord |
|---|---|---|---|---|
| head_foreground | +1.56 [+1.03, +2.09] | -0.01 [-0.02, -0.01] | +0.99 [+0.60, +1.38] | +0.00 [-0.00, +0.01] |
| head_supported | -0.14 [-0.34, +0.04] | -0.01 [-0.02, -0.01] | +0.10 [-0.10, +0.28] | -0.00 [-0.01, +0.01] |
| head_unsupported | +2.31 [+1.60, +3.00] | +0.00 [-0.00, +0.01] | +1.12 [+0.51, +1.69] | +0.01 [+0.01, +0.02] |

#### adaptive_r75


| region | PSNR vs hard | LPIPS utility vs hard | PSNR above chord | LPIPS utility above chord |
|---|---|---|---|---|
| head_foreground | +1.99 [+1.15, +2.85] | -0.04 [-0.05, -0.03] | +0.78 [+0.33, +1.25] | -0.01 [-0.02, -0.00] |
| head_supported | -1.00 [-1.31, -0.70] | -0.04 [-0.05, -0.03] | -0.51 [-0.68, -0.34] | -0.01 [-0.02, -0.01] |
| head_unsupported | +3.88 [+2.61, +5.13] | -0.02 [-0.04, -0.01] | +1.41 [+0.65, +2.14] | -0.00 [-0.02, +0.01] |

#### adaptive_r150


| region | PSNR vs hard | LPIPS utility vs hard | PSNR above chord | LPIPS utility above chord |
|---|---|---|---|---|
| head_foreground | +1.97 [+1.16, +2.79] | -0.05 [-0.07, -0.04] | -0.08 [-0.33, +0.15] | -0.01 [-0.01, -0.00] |
| head_supported | -1.19 [-1.53, -0.88] | -0.06 [-0.07, -0.04] | -0.44 [-0.62, -0.26] | -0.01 [-0.01, -0.00] |
| head_unsupported | +4.46 [+3.05, +5.93] | -0.04 [-0.06, -0.03] | +0.21 [-0.28, +0.72] | -0.01 [-0.02, -0.00] |
