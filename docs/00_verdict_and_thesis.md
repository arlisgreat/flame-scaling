# 研究判断与中心论点

## 一句话判断

FLAME 既可能是捷径，也可能是上限；两者并不矛盾。正确问题不是“要不要 FLAME”，而是对哪个作用、哪个区域、在多大的数据和模型预算下，FLAME 的估计误差收益开始小于它引入的近似误差。

## 可检验的理论框架

对方法 `m`、评估区域 `r`、训练目标区域 `T`、身份规模 `N` 和容量 `C`，把期望误差写成：

```text
Risk(m,r,T,N,C)
  = approximation_error(m,r)
  + estimation_error(m,r,T,N,C)
  + allocation_error(m,r,T,C)
  + supervision_error(m,r)
  + optimization_error(m,r,T,N,C)
```

硬 FLAME 通常降低小样本下的估计误差，但会在舌头、牙齿、头发、胡须、动态皱纹、饰品和强非刚性区域引入不可消失的近似误差。tracker 误差属于监督误差，不能被偷换成表示上限。oracle-fit 负责尽量压低估计误差与监督误差，训练 scaling 负责观察真实学习曲线。

### 3DGS 中的新增识别条件

`FLAME geometry` 不能只用 Gaussian mean 是否贴模板来定义。当前 oracle 实验证明，learnable covariance/extent 能在 mean 固定时充当隐藏的 geometry release。因此后续用：

```text
G = (G_mu, G_Sigma, G_density)
```

分别表示中心支撑、协方差/方向尺度与 densification/cardinality。H2 的 ceiling 证书必须明确控制这三项，否则 hard/free gap 不是可识别的单一 FLAME 效应。

## 三个主假设与证伪条件

### H1：小数据区间硬先验更高效

在预注册的低数据条件下，硬 FLAME 在相同模型与优化预算下，对核心面部区域的测试误差优于自由表示。

证伪：低数据下优势不稳定、只出现在 tracker 自评指标中，或在参数/优化预算匹配后消失。

当前证据：对 `A` 角色、五身份、8 training frames，free−hard oral 为 `−4.31 dB [−5.45,−3.37]`，0/5 为正，支持 H1 的 **per-identity observation-scale** 版本。跨身份 `G_mu` 实验则修正了原假设：full-target、`N_id=8` 时 head-supported/oral/LPIPS 偏好 hard，但 head-unsupported 已偏好 release；切换为 head-only target 后，released 的 head-foreground PSNR 已为 `+0.57 dB [0.23,0.94]`。所以 H1 只能限定为 `role × region × target × metric` 的 sample-efficiency prior，不能写成全局 hard 优势。

### H2：硬先验随规模扩大出现区域性平台

硬 FLAME 的高数据、高容量渐近误差在至少一个预注册非 FLAME 区域显著劣于 relaxed/free 表示，并且同一方向的差距在 oracle-fit 中存在。

证伪：更充分的优化消除差距；差距只来自 tracker 噪声；或所有观察区域的置信区间均兼容“无上限差异”。

当前证据：同一 A renderer 从 8 帧扩到 full sequence 后，free−hard oral 反转为 `+3.69 dB [2.83,4.73]`，5/5 为正。跨身份 `G_mu` 代理中，hard head-unsupported 从 `N_id=8` 到 384 基本保持约 5.7 dB，released-hard gap 从 `+1.51` 扩到 `+5.12 dB`；60k 使 gap 增至 `+5.78 dB`，2× density 后仍为 `+4.79 dB`，head-only target 后仍为 `+4.92 dB`。静态 G oracle 进一步控制 `G_mu/G_Sigma/G_density`：4× density 时 hard unsupported center error 仍为 `35.84 mm [31.17,39.97]`，而 free 为 `5.20 mm [4.84,5.57]`。这些共同排除简单欠优化、torso target 与 surface sampling density 解释。

### H3：APR 得到更好的 Pareto 前沿

区域和样本相关的先验释放，在相同总参数、训练 token、Gaussian 数和控制输入下，相对 hard、free 与固定 residual 基线形成非支配解。

证伪：收益来自额外参数；统一 release schedule 同样有效；或图像指标提高但几何、一致性和控制性下降。

当前证据：A-only spatial gate prototype 在 full oral 比 hard `+5.26 dB`、比 free `+1.57 dB`，5/5；但 whole-head 由 free 领先 `+7.16 dB`。跨身份 `G_mu` 中，adaptive r=30 在 `N_id=384` 仅用约四分之一 released displacement，相对 hard 提高 head-foreground `+1.56 dB`、unsupported `+2.31 dB`，且高于 hard→released displacement chord `+0.99/+1.12 dB`；supported PSNR 与 chord tie。它对 hard LPIPS 仍略差、对 released LPIPS 明显更好，因此验证的是局部 Pareto knee，不是全指标支配。完整 evidence-conditioned `G/C/P` release 仍未完成。

## 论文真正的新颖性

不主张以下内容本身新颖：template Gaussian、FLAME residual、local offset、学习 deformation、增加数据、增加容量、从 tracker 转向端到端估计。这些均有强先例。

论文应主张并证明：

1. **Causal factorization**：FLAME 的 `G/C/A/P` 四个作用可独立干预；
2. **Prior phase diagram**：交叉点是作用和区域相关的，不存在单一“FLAME 阈值”；
3. **Ceiling certificate**：oracle-fit 证明部分性能平台来自表示类而非训练失败；
4. **Evidence-adaptive release**：APR 学到何处、何时、释放哪一种作用，并与测得的 phase diagram 对齐；
5. **Reproducible protocol**：公开 split、区域、yaw/expression stress set、run ledger 与置信区间。

## 目标论文结构

- Finding 1：硬 FLAME 的 sample-efficiency 随 `N_id`、每身份观测密度和容量变化；
- Finding 2：不同区域的 crossover 顺序不同，舌/口内/头发等先于稳定皮肤区域；
- Finding 3：`G/C/A/P` 的效应不同，tracker 污染不能解释全部 ceiling；
- Method：APR 以局部证据控制先验精度，统一覆盖 hard 到 free；
- Benchmark：公共、多视图、区域级、支持/不支持区域并存的评测协议。

推荐题目：**FLAME: Shortcut or Ceiling? A Causal Scaling Study of Parametric Head Priors**。

## 当前最重要的反直觉合成结论

1. hard FLAME means 可以通过大 covariance “画出”模板外内容，而不修复 center geometry；
2. free A 随观测规模从显著劣于 hard 变成显著优于 hard，但过大 capacity 会再次恶化；
3. adaptive A 提高 clean 上限，却对 tracking lag 比 hard 更敏感；
4. correspondence 不必全局精确到唯一 vertex 才有价值，10 mm 邻域 local identity 保留了大量收益，但仍不是免费的；
5. 因而 `release representation`、`retain local semantics` 和 `correct pseudo-labels` 是三个不同操作，不能用一个 residual 分支统称。
6. 增加 FLAME-surface Gaussian 密度并不会消除 geometry ceiling；它主要在宽 covariance 下给 hard 模型更多“画到模板外”的机会，因此更高 RGB 容量甚至会掩盖更稳定的中心几何错误。
7. full-foreground target 会把 released slots 大量 transport 到 neck/torso；切到 head-only 后 neck displacement 从约 101 mm 降到 20 mm，supported PSNR 与 LPIPS penalty 基本消失，但 unsupported gain 保留。训练目标区域不是评估细节，而是 prior allocation 的因果变量。
