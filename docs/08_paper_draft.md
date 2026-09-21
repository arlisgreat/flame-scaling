# FLAME: Shortcut or Ceiling? A Causal Scaling Study of Parametric Head Priors

Working paper draft, 16 July 2026. All numbers in the main text below correspond to completed experiments. Post-primary analyses are labeled explicitly. The intended contribution is a causal empirical study; a full joint `G/C/A/P` release architecture is not claimed as completed.

## Abstract

Parametric head models are used simultaneously as geometry, correspondence, animation, and supervision priors in modern Gaussian avatars. This coupling makes a basic question unanswerable from existing method comparisons: does FLAME improve learning by providing a sample-efficient shortcut, or limit asymptotic quality by restricting the representation? We factor FLAME into canonical geometry `G`, semantic correspondence `C`, animation `A`, and tracker pseudo-labels `P`, and further decompose Gaussian geometry into mean support `G_mu`, covariance `G_Sigma`, and density `G_density`. Across static multiview oracles, dynamic held-out-frame experiments, and a 415-identity public NeRSemble study, we find that the answer is conditional rather than binary. Hard animation is 4.31 dB better than learned animation with eight training frames, but becomes 3.69 dB worse with full sequences. In cross-identity reconstruction, hard FLAME remains nearly flat on head content outside its support, whereas the released-mean advantage grows from 1.51 dB at eight identities to 5.12 dB at 384 identities. This gap survives 60k-step convergence, decoder-capacity controls, a head-only objective, and twice as many FLAME-surface Gaussians. However, hard anchoring preserves supported-region sharpness under a full-foreground objective; changing the target to the head alone reduces released-query neck displacement from 101 to 20 mm, reverses the supported-region PSNR gap, and removes the perceptual penalty. Covariance can also reverse the relative ordering without improving absolute quality, exposing a geometric bypass in image-space metrics. Finally, a 30 mm adaptive release attains head quality above the hard-to-released displacement chord at large identity scale, but does not dominate hard FLAME in LPIPS. These results support a region-, metric-, and evidence-conditioned prior phase diagram rather than a global FLAME threshold.

## 1. Introduction

Template priors solve a real statistical problem. A FLAME vertex already has a canonical position, a semantic identity, blendshape motion, skinning weights, and a tracker-derived control trajectory. Predicting Gaussian attributes on this scaffold is easier than predicting an unordered, controllable 3D head from images. GGHead, LAM, PanoLAM, and recent large avatar systems exploit this regularity. Conversely, Portrait4D-v2, LVSM, MVCHead, Any3DAvatar, and FastGHA demonstrate that learned or weakly structured representations can replace parts of the conventional 3D prior when enough data or supervision is available.

These systems do not isolate the source of the gain. Replacing FLAME commonly changes the query coordinates, correspondence, deformation rule, control labels, architecture, and training data at once. A residual offset is also not a sufficient causal test: Gaussian covariance can stretch anchored splats beyond the mesh, increased density can amplify this shortcut, and a full-foreground loss can reward reallocating head queries to clothing and shoulders. Consequently, a reported image-quality gain may reflect better geometry, a rendering bypass, a different allocation of a finite point budget, or tracker correction.

We study the question in a shared representation with explicit intervention interfaces. Our central thesis is:

> FLAME is a sample-efficient allocation prior on supported, semantically stable regions and a representation bottleneck outside its support. The transition is not a global data threshold; it is a phase boundary over role, region, training target, representation channel, capacity, and metric.

Our contributions are:

1. We introduce a causal factorization `G/C/A/P` and identify `G_mu/G_Sigma/G_density` as separately controllable Gaussian geometry channels.
2. We provide an oracle ceiling certificate showing that covariance and density can improve RGB without repairing center geometry, while the FLAME support error remains.
3. We measure both observation-scale and identity-scale phase changes with paired seeds and held-out identities, including capacity, convergence, covariance, target-region, and density controls.
4. We identify a limited adaptive-release Pareto knee and preserve the negative results: no global crossover, no universal small-data hard advantage, and no all-metric APR dominance.

## 2. Related Work and Novelty Boundary

Template Gaussian avatars use regular geometry or UV layouts to reduce prediction difficulty. GGHead shows that template regularity and sampling density can matter as much as exact FLAME shape. LAM uses FLAME canonical vertices as Gaussian queries and inherits correspondence and animation, while explicitly noting tongue, dynamic-wrinkle, and tracking limitations. PanoLAM retains sparse FLAME points before densification. NPGA, OMEGA-Avatar, FFAvatar-Y, SVG-Head, and PhysHead already contain residual deformation or region-specific representations. MATCH learns correspondence with a fixed template. Therefore, local offsets, learned correspondence, and hybrid head/hair representations are not our novelty.

At the other endpoint, Portrait4D-v2 avoids inaccurate 3DMM reconstruction but requires precise synthetic multiview data; LVSM observes scale-driven generalization with minimal explicit 3D bias; MVCHead and FastGHA learn freer Gaussian distributions or dynamics. HeadsUp is the closest scaling-study precedent and examines identity count, input views, capacity, and neutral templates. Our remaining novelty is narrower: role-level interventions in a shared renderer, a geometry-channel ceiling certificate, regional crossover curves, and target/density controls that distinguish representation support from finite-budget allocation.

## 3. Causal Variables

We represent FLAME use by `(G,C,A,P)`:

- `G`: canonical query geometry. We intervene on `G_mu` while separately fixing or scanning covariance `G_Sigma` and point count `G_density`.
- `C`: FLAME-semantic attribute identity. A center-preserving shuffle identifies its effect independently of geometry.
- `A`: deformation from FLAME blendshapes and LBS versus bounded residual or learned deformation under the same control code.
- `P`: tracker-derived pose, expression, and camera. Controlled temporal lag changes `P` while retaining clean RGB, alpha, and evaluation regions.

For target region `T`, evaluation region `r`, identity scale `N`, and capacity `K`, the relevant risk is

```text
Risk(m,r,T,N,K)
  = approximation error(m,r)
  + estimation error(m,r,T,N,K)
  + allocation error(m,r,T,K)
  + supervision error(m,r)
  + optimization error(m,r,T,N,K).
```

The allocation term is essential for finite Gaussian sets: released centers may cover unsupported content by removing points from supported detail.

## 4. Experimental Design

### 4.1 Static oracle ceiling study

We use five multiview NeRSemble cases with landmark-anchored, visually audited FLAME fits. Per-instance optimization crosses hard/free means, 3/30 mm Gaussian-axis caps, and 0.25x/1x/4x nested densities. RGB is reported with center-to-target geometry. Held-out angular folds test whether train-view improvements generalize.

### 4.2 Dynamic `A/C/P` identification

Five identities and four motion families use official FLAME 2023 tracking. For `A`, hard FLAME motion, a bounded spatial residual, and learned deformation receive the same tracking code and renderer. Training frames are nested. For `C`, centers remain fixed while attributes are shuffled locally or globally. For `P`, expression, pose, and rigid transforms receive controlled 1/2/4-frame lag while image targets and regions remain clean.

### 4.3 Cross-identity `G_mu` scaling

We cache 415 usable NeRSemble FREE identities. Nested non-benchmark training sets contain 8, 32, 128, and 384 identities; seven identities are reserved for validation and 20 official SVFR identities for test. Each identity uses one static frame and 15 common calibrated cameras. Models observe nested prefixes of 1, 4, or 8 cameras and predict only excluded cameras.

Every condition uses 5,023 semantic slots, a shared encoder/decoder, 651,816 trainable parameters, fixed 5 mm isotropic covariance, 20k steps, and three paired seeds. The sole primary intervention is `G_mu`: hard zero displacement, adaptive gated displacement, or released bounded displacement. The identity-specific scaffold is fit with multiview landmarks, including target-camera geometry, deliberately favoring hard FLAME; target RGB never enters the encoder or slot colors. This is a static mechanism proxy, not a deployable single-view avatar system.

We average target cameras within identity, then average paired seed effects within identity, and bootstrap the 20 test identities 10,000 times. The broad unsupported mask was pre-registered. After qualitative audit revealed substantial clothing and shoulders, we retained it unchanged and added a disclosed frozen-checkpoint head ROI: projected face, oral, eye, scalp, and ear vertices, excluding neck, dilated by six pixels.

### 4.4 Post-primary controls

Before observing each control result, we froze: 3 mm covariance; hidden dimensions 64 and 512; 60k convergence; adaptive radii 30 and 75 mm; a head-only training target; and 2x density with 5,023 additional topology-consistent surface samples. All controls retain the same held-out identities and paired seeds.

## 5. Results

### 5.1 Gaussian covariance is a hidden geometry-release channel

In the five-case static oracle, increasing the extent cap from 3 to 30 mm improves hard unsupported RGB by 10.94 dB, while the hard center error remains fixed. The mean-by-extent interaction is 1.79 dB with all five cases positive. Increasing density from 0.25x to 4x improves hard/3 mm by only 0.52 dB but hard/30 mm by 3.82 dB; the density-by-extent interaction is 3.30 dB. At 4x, hard unsupported center error remains 35.84 mm, compared with 5.20 mm for free centers. More or larger splats can hide a stable support error.

### 5.2 Animation changes sign with observation scale

With eight training frames, learned `A` is 4.31 dB worse than hard FLAME in the oral region for all five identities. With full sequences, it is 3.69 dB better for all five. Adaptive motion is 5.26 dB above hard and 1.57 dB above free in the oral region, but free remains 7.16 dB better over the whole head. Increasing deformation rank hurts low-data generalization and yields an intermediate optimum at full scale. This is a regional bias-variance transition, not monotonic benefit from model size.

Correspondence and tracking are independent. Local/global attribute shuffles preserve every center but reduce target quality, with local shuffles less harmful than global ones. A four-frame tracking lag reduces target-region hard quality by 1.52--2.85 dB across head, eye, mouth, and jaw motion. Adaptive motion drops more from its clean baseline than hard motion, so representation release cannot substitute for tracker correction.

### 5.3 Identity scaling exposes a hard support plateau

At four observed cameras, hard full-frame PSNR changes only from 11.25 to 11.37 dB between 8 and 384 training identities; released changes from 12.33 to 14.73 dB. The released-hard advantage grows from 1.09 to 3.37 dB. In the post-audit head ROI:

| Region | `N_id=8` released-hard | `N_id=384` released-hard |
|---|---:|---:|
| head foreground PSNR | -0.09 dB, CI crosses zero | +2.46 dB |
| head supported PSNR | -2.37 dB | -0.89 dB |
| head unsupported PSNR | +1.51 dB | +5.12 dB |
| torso outside head PSNR | +2.89 dB | +7.09 dB |

The unsupported advantage is already positive at the lowest scale and grows by 3.61 dB, so its crossover is left-censored. Supported and oral quality do not cross by 384 identities. Spatial LPIPS favors hard FLAME on all head regions under the full-foreground objective. The observed phase change therefore depends on both region and metric.

### 5.4 Capacity and convergence reinforce the result; covariance moves it

Hidden dimensions 64, 128, and 512 retain the same ordering. At `N_id=384`, the released unsupported advantage is 4.95, 5.12, and 5.60 dB. At 60k steps, hard unsupported quality remains unchanged while the released advantage increases to 5.78 dB. The gap is not a 20k optimization artifact.

Covariance changes the boundary. With 3 mm splats, released becomes 1.32 dB better on the supported region and better in relative LPIPS; with 5 mm splats it is 0.89 dB worse and worse in LPIPS. Yet both 3 mm methods have much worse absolute LPIPS. Narrow splats expose anchored sampling holes earlier; they do not improve the system. A `G_mu` scaling claim is therefore incomplete without `G_Sigma`.

### 5.5 Target region causally controls finite-slot allocation

Switching only the training loss from full foreground to the head leaves hard performance effectively unchanged. Released head-foreground and supported PSNR improve by 0.87 and 1.46 dB, while torso PSNR decreases by 7.36 dB. Released mean displacement falls from 32.34 to 14.23 mm; neck displacement falls from 101.22 to 19.86 mm.

Under the head-only target at `N_id=384`, released is 3.34 dB better on head foreground, 0.57 dB better on supported head, and 4.92 dB better on unsupported head. The three seed effects are directionally identical. LPIPS differences on all head regions become statistically compatible with zero. Thus the primary supported/perceptual penalty is mainly an objective-induced allocation conflict. The unsupported gain survives removal of the torso reward and is not a torso artifact.

### 5.6 Twice as many surface queries do not remove the support ceiling

At `N_id=384`, 2x density improves supported PSNR for both methods and reduces the released deficit from 0.89 to 0.48 dB. It also approximately halves the head LPIPS gap. Nevertheless, the released unsupported advantage remains 4.79 dB across all three seeds. At `N_id=8`, density does not move the crossover left; the unsupported advantage decreases from 1.51 to 0.91 dB. More FLAME-surface samples reduce sampling and allocation pressure but cannot create support outside the surface.

### 5.7 Adaptive release finds a data-dependent Pareto knee

At `N_id=384`, adaptive 30 mm release uses 7.62 mm mean displacement versus 32.34 mm for the released endpoint. Relative to hard it improves head-foreground by 1.56 dB and unsupported head by 2.31 dB, while the supported difference is -0.14 dB with a confidence interval crossing zero. At each identity's actual displacement, it lies 0.99 dB above the hard-to-released chord on head foreground and 1.12 dB above it on unsupported head. It is approximately tied with the chord on supported head.

This is a local Pareto improvement, not dominance. Adaptive 30 mm is 0.012--0.014 worse than hard in head LPIPS, although about 0.045 better than released. At eight identities it does not improve head-foreground PSNR. Adaptive prior release itself has a data-dependent phase boundary.

## 6. Discussion

The results distinguish three mechanisms that are often collapsed into one residual branch:

1. **Representation support:** hard centers cannot populate content absent from the template. This ceiling persists with data, convergence, capacity, target changes, and more surface points.
2. **Finite-budget allocation:** released slots can cover unsupported regions at the expense of supported density. The training target determines the exchange rate.
3. **Rendering bypass:** covariance and cardinality can improve image metrics without repairing centers, obscuring the representation error.

The practical implication is not to remove FLAME everywhere. Stable supported regions benefit from semantic density and low-variance motion, especially with limited observations. Unsupported regions require releasable support; tracker correction remains separate. An effective system should condition prior precision on independent multiview conflict, regional support, motion residual, and tracker confidence, and should report its behavior jointly over geometry and perceptual metrics.

## 7. Limitations

The cross-identity study is a 128x88 static proxy. Its FLAME scaffold uses multiview landmarks, including target-camera geometry, and therefore cannot establish deployable single-view performance. The 20 official test identities support paired inference for this public protocol but not a universal exponent or asymptotic threshold. The head ROI is post-audit, though all checkpoint evaluations are frozen and the broad pre-registered result is retained. The target and density controls are post-primary and labeled as such. Dynamic public data provide only a single training camera, so animation results are temporal held-out-frame rather than novel-view. Natural tracker-error distributions, learned local correspondence transport, dynamic multiview geometry, and a joint evidence-conditioned `G/C/A/P` APR remain open.

## 8. Reproducibility Checklist

- 415 usable cached identities; nested train sets `8/32/128/384`; seven validation and 20 official test identities.
- Three paired seeds per primary/control cell; camera aggregation before identity bootstrap.
- 36 primary, 75 convergence/capacity/covariance/dose, 18 head-target, and 18 density2 training runs.
- Frozen checkpoint head-ROI evaluations with source metrics/checkpoint SHA verification.
- Primary and post-primary protocols stored under `artifacts/audits/`.
- Full result tables under `artifacts/analysis/identity_scaling20k_v1/`, `identity_scaling20k_v1_head_roi/`, `identity_scaling_followups_v1/`, and `identity_scaling_mechanism_controls_v1/`.

## Appendix A. Claim Status

| Claim | Status |
|---|---|
| `G_mu/G_Sigma/G_density` must be separated | Supported by oracle and population covariance/density controls |
| Hard support forms an unsupported-region ceiling | Supported in static oracle and public identity scaling; dynamic multiview transfer remains open |
| A single global FLAME crossover exists | Rejected |
| Small-data hard is universally better | Rejected; only region/metric/target-conditioned variants survive |
| More capacity or optimization removes the gap | Rejected over tested ranges |
| More surface points remove the gap | Rejected at 2x population density and 4x oracle density |
| Full-target released penalty is intrinsic | Rejected by head-only target control |
| Adaptive release dominates hard and free | Rejected; a local PSNR/displacement Pareto knee is supported |
| Full joint evidence-conditioned APR is complete | Not claimed |
