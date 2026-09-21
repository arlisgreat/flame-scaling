# 统计分析与报告规则

## 分析单位

主分析单位是 `(split, seed, N_id, N_obs, capacity, method, region, yaw_bin, sequence_family)`。同一 seed/split 的 hard、free、APR 必须配对，禁止把每一帧误当作独立样本来虚增显著性。

帧和相机作为层级内重复测量；identity、sequence 和 seed 进入 random effects。对 test identity 先聚合，再做跨 identity 推断。

## Scaling law

先拟合可解释的饱和模型，再用非参数曲线复核：

```text
E(m,r,N,C) = E_inf(m,r)
           + a(m,r) * N^(-alpha(m,r))
           + b(m,r) * C^(-beta(m,r))
           + d(m,r) * (N*C)^(-gamma(m,r))
```

身份数 `N_id` 与每身份观测 `N_obs` 分开拟合。若参数不可识别，不强行报告 scaling exponent，退回单调 spline / isotonic bootstrap 和观测范围内的区间结论。

## Crossover 定义

对 lower-is-better metric 定义：

```text
Delta_r(N,C) = E_hard(r,N,C) - E_free(r,N,C)
```

- `Delta < 0`：hard 更好，支持 sample efficiency；
- `Delta > 0`：free 更好，支持 release；
- crossover `N*`：配对差值的期望第一次由负变正的位置。

报告 `N*` 的 cluster bootstrap 95% CI。若观测范围内不交叉，写成左删失/右删失，不外推一个虚假点估计。APR 分别与 hard、free 做相同分析。

## 假设检验

### H1 主终点

低数据 `N_id=32`、`N_obs=16`、容量 `M`，stable-skin 与 full-face 的 paired LPIPS/DISTS。hard 相对 free 的优势需在预注册聚合指标上置信区间不跨零，并在至少两个 seed/split 复现。

### H2 主终点

高数据、高容量条件下，`tongue+oral`、`hair`、`dynamic-detail` 的 hard-free gap；同时要求 oracle-fit gap 同方向。比较渐近项 `E_inf` 时提供 profile/bootstrap interval。

### H3 主终点

APR 的 Pareto dominance probability：在 bootstrap sample 中，APR 同时不劣于 hard/free 的质量、geometry、control，并至少在一项严格更优的概率。

## 多重比较与效应量

- 三个预注册区域组和三个主假设构成 confirmatory family；Benjamini-Hochberg FDR `q=0.05`；
- 其他细分区域、yaw、expression 是 exploratory，明确标注；
- 同时报告绝对差、相对差、标准化效应和 CI，不能只报 p-value；
- 所有主图显示 seed/split 点和不确定性，不只画平滑均值。

## 机制验证

APR 的 gate 需要通过以下独立相关性检验：

- `rho_G` 与 oracle geometry gap；
- `rho_C` 与 semantic transport entropy / cross-view conflict；
- `rho_A` 与非 FLAME motion residual；
- `rho_P` 与 tracker confidence / self-cal correction。

用 held-out identity 计算，防止用训练 residual 自证。gate 可视化是机制证据，不替代质量实验。

## 报告纪律

- checkpoint selection 只看 validation，不看 test；
- 失败 run 不得静默删除，ledger 记录原因和重跑规则；
- 同时报告 equal-step 与 converged-budget；
- 数据、参数、Gaussian 数、render resolution、训练 token 和实际 GPU-hours 全部入表；
- 对所有不支持 H1/H2/H3 的结果保留并解释，不把论文写成只有正结果。

## NeRSemble 静态 `N_id` proxy 的预注册推断

该 proxy 使用一个固定 identity split 和 3 个配对训练 seed `{0,1,2}`，仍是静态机制证据，不升级为最终动态 confirmatory scaling law。主要 effect 定义为同一 test identity、同一 seed、同一 `N_id/N_obs/region` 下 `adaptive/released` 相对 hard 的差；camera 先平均，不把 14 个 target views 当独立样本。主 CI 先在同一 identity 内平均三个 seed 的 paired effect，再对 20 个固定 SVFR identity 做 cluster bootstrap 95% CI（10k，bootstrap seed `20260715`）；另列每个训练 seed 的 identity-clustered effect，禁止把 60 个 `identity × seed` 伪装成独立样本。

预先报告三类量：每个 cell 的绝对 PSNR/MAE；`release − hard` 的 identity-paired advantage；以及 `N_id=384` 与 `N_id=8` 之间 release advantage 的差分中的差分。观测点只允许报告 observed、left-censored 或 right-censored crossover，不拟合超出 `{8,32,128,384}` 的阈值。checkpoint 只由 7 个固定 validation identity 的 foreground MAE 选择。

该 proxy 的优先区域是 `FLAME-unsupported` 与 oral，full frame 作为汇总，eyes 为 exploratory。由于 multiview landmark scaffold 使用 target camera 几何信息，图像指标可用于比较 mean release，但不能解释为 deployable single-view reconstruction 精度；必须同时披露这一对 hard 有利的 oracle 偏置。
