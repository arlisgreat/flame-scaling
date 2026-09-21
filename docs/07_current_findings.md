# 当前发现、反直觉结果与证据等级

更新时间：2026-07-16。本页只总结已经实际运行并通过视觉审计的实验，不把计划写成结果。

## F0：可信 scaffold 是必要前提

最初使用 point-cloud ICP 加弱 coefficient prior 的 FLAME 拟合虽有较低表面距离，但 coefficient `L2≈49`、最大绝对值约 15，且出现 nose/lip 语义漂移。由它得到的旧五案例曲线只能作诊断，不能作论文证据。

当前替代方案使用 13/16 个标定视图中的稳定 MediaPipe anchor（双眼、鼻尖、嘴角、上唇）做鲁棒三角化，再以 coefficient `±3 sigma`、shape/expression prior 和排除 mouth neighborhood 的表面 ICP 拟合。五案例 coefficient L2 为 `6.13–9.82`，语义最大误差为 `1.32–1.87 mm`；overlay 已逐例审计。伸舌帧还明确排除会被舌尖吸引的 chin/lower-lip landmark。

## F1：mean release 是连续剂量，不是 hard/free 二元开关

Subject 017、`EXP-6-tongue-1`、frame 182，16 个标定视图，5,023 Gaussians，4,000-step all-view oracle：

| mean release | supported PSNR | unsupported PSNR | unsupported center error |
|---:|---:|---:|---:|
| 0 mm | 23.58 | 11.74 | 44.97 mm |
| 25 mm | 33.63 | 19.42 | 32.29 mm |
| 50 mm | 35.82 | 28.90 | 22.35 mm |
| 75 mm | 36.14 | 33.24 | 18.87 mm |
| 100 mm | 36.24 | 35.08 | 17.18 mm |
| 150 mm | 36.42 | 36.10 | 16.01 mm |
| 200 mm | 36.36 | 35.90 | 17.13 mm |
| free | 36.39 | 36.25 | 18.97 mm |

unsupported 区域的恢复主要发生在 25–100 mm，200 mm 已接近 free。这支持“约束强度曲线”而不是“是否使用 FLAME”的表述。

但该区域包含头发、衣领、颈肩和 oral protrusion，不能把 +24.51 dB 全部归因于舌头。

## F2：反直觉负结果——hard FLAME means 仍能画出舌头

经过逐视图审计的 oral-protrusion polygon 中，30 mm Gaussian axis cap 下，free 只比 hard 高 **+1.15 dB**。这否定了“只要中心固定在 FLAME 上，伸舌 RGB 就必然失败”的简单命题。

原因是 Gaussian covariance 本身提供第二条几何通道。原 hard 条件的 mouth-neighborhood 最大轴标准差 p90 为 **14.41 mm**；三倍标准差约 43 mm，少量 Gaussian 还达到 30 mm 标准差上限。它们能从错误的中心位置伸展到舌头/颈肩区域，用模糊、方向性 splat 改善 RGB，而不修复中心几何。

## F3：`G_mu × G_Sigma` 因子实验识别出 covariance bypass

同一伸舌帧，mean constraint 取 `{hard, 50 mm, free}`，Gaussian 最大轴标准差取 `{3,6,12,30} mm`，其余参数量、优化器、损失和 4,000-step 预算相同：

- oral ROI 的 free-hard gap：3 mm cap 为 **+4.67 dB**，30 mm cap 为 **+1.15 dB**；
- 只把 extent cap 从 3 放到 30 mm，hard 提升 **+5.46 dB**，free 只提升 **+1.93 dB**；
- extent-gain difference-in-differences 为 **+3.53 dB**；
- hard 的 unsupported center error 在所有 cap 下固定为 **44.97 mm**，但 RGB 大幅改善。

这直接证明 image quality 与 center geometry 可以解耦。只扫描 mean offset 的实验不能唯一识别“FLAME geometry bias”。

## F4：covariance bypass 在五案例中复现

五个 NeRSemble NVS 真人案例，可信 landmark-anchored scaffold，配对 `hard/free means × 3/30 mm extent`：

| unsupported-region effect | mean | bootstrap 95% CI |
|---|---:|---:|
| hard: 30 mm minus 3 mm extent | +10.94 dB | [+9.73,+12.17] |
| free: 30 mm minus 3 mm extent | +9.14 dB | [+7.85,+10.42] |
| free minus hard at 3 mm extent | +19.21 dB | [+16.58,+21.86] |
| free minus hard at 30 mm extent | +17.42 dB | [+14.65,+20.64] |
| difference-in-differences | **+1.79 dB** | **[+0.60,+2.98]** |

interaction 在 5/5 案例为正。supported 和 whole-head 的 interaction 也分别为 +3.45 dB `[+1.33,+5.01]` 与 +2.62 dB `[+1.66,+3.59]`。

因此目前最稳的发现是：**FLAME mean anchoring 的测得收益取决于 Gaussian extent capacity；硬 mean 可通过 covariance 部分越狱，导致 RGB 指标低估表示上限并掩盖错误几何。**

## F5：held-out view 揭示区域条件化的 bias–variance trade-off

在 subject 017 上把 16 个相机按角度分成四折，每个相机恰好 held out 一次；每折用其余 12 个相机优化完整 `{0,10,25,50,75,100,150,200,free}` mean-release ladder：

- oral held-out 由 75 mm 最优，27.11 dB；比 hard 高 **+1.54 dB**，camera-bootstrap 95% CI `[+0.26,+2.91]`；比 free 高 +0.63 dB `[-0.02,+1.36]`；
- broader unsupported held-out 由 free 最优，27.62 dB；比 oral-optimal 75 mm 高 **+0.55 dB** `[+0.08,+0.98]`；
- oral/unsupported 的非支配条件为 `75 mm, 200 mm, free`，不存在单个全局最优 release。

这说明 hard 是低方差但高偏差，free 是高容量但更容易产生 train–held-out gap；中间 release 可在 FLAME-supported/oral 区域提供正则化，而外部非 FLAME 内容需要更自由的支撑。

为避免该结论只来自选择 75 mm 的 tongue case，又在五个独立 NeRSemble NVS subject 上做三折 angular cross-validation，比较预先选定的 `{hard,75 mm,free}`：

- supported：75 mm 比 hard **+7.10 dB** `[+4.69,+9.84]`，比 free **+0.13 dB** `[+0.01,+0.24]`；4/5 subject 由 75 mm 最优；
- unsupported：75 mm 比 free **−0.45 dB** `[-0.89,−0.00]`；4/5 subject 由 free 最优；
- free 的 train–held-out gap 始终大于 75 mm，方向与“bounded release 提供正则化”一致。

这是当前首个可复现的区域 prior phase diagram 证据：**同一数据、模型和优化预算下，先验强度的最优值随区域改变。** 它仍是 per-instance held-out-view 研究，不是跨身份训练 scaling law。

## F6：动画偏置出现可复现的 observation-scale crossover

使用 NeRSemble Mono FLAME Avatar 的官方 `flame2023_tracking_v2`，先复现官方 FLAME 2023 前向：shape/expression、neck/jaw/eyes LBS 在 model space 中计算，再施加分离的全局 rotation/translation/scale。五个时刻的 mesh-to-image overlay 已视觉审计。动态因果契约固定：

- `G`：5,023 个 FLAME canonical queries；
- `C`：固定 vertex identity；
- `P`：官方 tracking v2；
- renderer、Gaussian 数、总参数和测试帧一致；
- 只切换 `A-hard / A-adaptive / A-free`。

`A-hard` 使用 FLAME expression + neck/jaw/eye LBS；`A-free` 从 neutral identity mesh 学习 tracking-code-conditioned deformation；`A-adaptive` 在 hard A 上学习带 release penalty 的逐顶点 30 mm bounded residual。测试帧每五帧固定 held out 一帧。五身份 `EXP-5-mouth` 的配对结果：

| oral held-out effect | mean | identity-bootstrap 95% CI | direction |
|---|---:|---:|---:|
| free − hard，8 training frames | **−4.31 dB** | `[−5.45,−3.37]` | 0/5 positive |
| free − hard，full sequence（142–183 train frames） | **+3.69 dB** | `[+2.83,+4.73]` | 5/5 positive |
| adaptive − hard，full | **+5.26 dB** | `[+4.41,+6.08]` | 5/5 positive |
| adaptive − free，full | **+1.57 dB** | `[+1.33,+1.79]` | 5/5 positive |

从 8 帧到 full，hard oral 只提高 `+1.36 dB [1.20,1.54]`，adaptive 提高 `+5.62 dB [4.90,6.46]`，free 提高 `+9.37 dB [7.49,11.57]`。这首次在共同 trainable renderer 和相同动态控制输入下观测到 hard/free 的符号反转。

但区域最优仍不同：full scale 时，whole-head 上 free 比 adaptive 高 **+7.16 dB** `[+3.58,+11.21]`，oral 上 adaptive 反而比 free 高 **+1.57 dB**。因此 crossover 不能压缩成单一全局阈值。

该结论是 **每身份观测帧数 scaling**，不是跨身份预训练的 `N_id` scaling；公开数据只有单训练相机，所以测试是 temporal held-out，不是 novel-view。

## F7：容量与数据发生强交互，释放不是越大越好

在同一五身份 mouth protocol 上扫描 learned deformation rank `K={4,8,16,32,64}`：

- 8 帧时，free oral 从 K=4 的 26.71 dB 单调降到 K=64 的 20.70 dB；K=4 比 K=64 高 **+6.01 dB**；
- free 的 oral train–held-out gap 从 K=4 的 **12.02 dB** `[10.78,13.15]` 增至 K=64 的 **17.07 dB** `[15.57,18.85]`；
- 8 帧下 adaptive 相对 hard 从 K=4 的 `+1.3 dB` 降到 K=64 的 `−0.3 dB`，free 从 `−1.9 dB` 降到 `−7.9 dB`；
- full scale 时，adaptive oral 在 K=32 最优，为 **35.66 dB** `[35.06,36.26]`；free 在 K=16 最优，为 **33.64 dB** `[32.97,34.30]`。

因此 capacity 既可能暴露表示上限，也可能扩大 estimation error。论文中的 phase diagram 必须是 `data × capacity × region`，不能只把模型做大后称为“free ceiling”。

## F8：动画收益与运动区域方向一致，但 mouth 有明确反例

五身份、固定 K=16、四个 motion families（head/eyes/mouth/jaw）上，以预注册 target-control 区域做 adaptive-gain difference-in-differences：

| sequence | target − control | full-scale interaction | identity-bootstrap 95% CI | positive subjects |
|---|---|---:|---:|---:|
| head | head − oral | **+2.10 dB** | `[+1.23,+2.95]` | 5/5 |
| eyes | eyes − oral | **+0.76 dB** | `[+0.20,+1.33]` | 5/5 |
| jaw | oral − eyes | **+1.13 dB** | `[+0.32,+2.26]` | 4/5 |
| mouth | oral − eyes | +0.21 dB | `[−0.47,+1.05]` | 2/5 |

前三个 family 支持 motion-aligned regional release；mouth family 没有显著 localization，禁止隐藏或改写为成功结果。它可能来自 mouth sequence 同时包含眼/脸联动，也可能说明当前 gate 只学到通用 tracking correction，而非纯区域机制。

## F9：P 误差是独立污染源，而且 adaptive A 更依赖准确 P

保持 clean RGB/alpha、clean evaluation ROI、`G/C/A` 不变，只将官方 tracking 的 expression、pose 和 rigid transform 整体错位 `1/2/4` video frames（约 41/82/165 ms）。full-scale hard A 在 4-frame lag 下的目标区域损失：

| motion/region | hard lag4 − clean | identity-bootstrap 95% CI |
|---|---:|---:|
| head/head | **−1.52 dB** | `[−1.77,−1.24]` |
| eyes/eyes | **−2.85 dB** | `[−3.44,−2.26]` |
| mouth/oral | **−2.41 dB** | `[−2.83,−1.98]` |
| jaw/oral | **−1.74 dB** | `[−2.12,−1.34]` |

四项均 5/5 下降，aggregate 剂量曲线随 lag 单调恶化。更多训练帧不能消除 hard-P 错位，因为 hard path 没有 active correction DoF。

更反直觉的是，adaptive 并没有“吸收”P 错位。以 lag2 比较各自 clean baseline，`adaptive drop − hard drop` 在上述四个目标区域分别为 **−0.95、−1.36、−0.85、−0.93 dB**，identity bootstrap CI 全部小于零、5/5 为负。adaptive 在 lagged 条件下通常仍优于 lagged hard，但它从自身 clean 上限掉得更多。这说明：**释放表示偏置提高了上限，却也提高了对伪标签时间精度的敏感性；A release 与 P correction 是互补项，不可互相替代。**

这只是受控 temporal lag，不是对自然 tracker error distribution 的估计。

## F10：固定 correspondence 的价值可在中心完全不变时独立测出

保持逐帧 Gaussian centers、FLAME animation、tracking、参数和优化预算完全相同，只把 color/scale/opacity/quaternion 的 query identity 做确定性逐帧重排：

- local：只在 10 mm canonical bins 内重排，实际改变比例约 `0.18/0.40/0.85`；
- global：全局重排 `0.25/0.50/1.00`。

五身份 full-scale 最大干预：

| sequence/target | local shuffle | global shuffle |
|---|---:|---:|
| eyes/eyes | **−4.08 dB** `[−4.74,−3.25]` | **−8.53 dB** `[−9.14,−7.79]` |
| mouth/oral | **−1.70 dB** `[−1.80,−1.59]` | **−7.47 dB** `[−8.10,−6.72]` |

所有差值均 5/5 为负。即使最小 local intervention 只改变约 18% identities，eyes/eyes 仍损失 **−2.47 dB**、mouth/oral 损失 **−1.04 dB**。更多数据能减轻但不能清零损失。

这证明固定 attribute identity 的作用不是 `G` 或 `A` 的替身；同时 local 显著好于 global，提示未来 `C-adaptive` 不必在 hard vertex ID 与完全无序之间二选一，局部 soft transport 是更合理的中间态。限制是：该结论针对 shared-attribute Gaussian oracle，不能外推为“无结构 Transformer 无法从图像推断隐式 correspondence”。

## F11：更多 FLAME-surface Gaussians 会放大 bypass，但不会移除支撑面 ceiling

在同一五案例静态 oracle 上补齐 `G_density={0.25×,1×,4×}`，与 `G_mu={hard,free}`、`G_Sigma cap={3,30} mm` 组成 2×2×3 因子实验。0.25× 使用 1,256 个嵌套 FPS FLAME vertices，1× 使用 5,023 个原始 vertices，4× 使用 20,092 个 points（保留全部 vertices，再在相同 FLAME 三角面上确定性面积采样）。每个 density 内 hard/free 的 Gaussian 数、外观参数、优化器、4,000 steps 和 13 个标定视图相同。

unsupported 区域的 0.25×→4× density gain：

| condition | PSNR gain | identity-bootstrap 95% CI |
|---|---:|---:|
| hard means / 3 mm cap | **+0.52 dB** | `[+0.15,+1.02]` |
| hard means / 30 mm cap | **+3.82 dB** | `[+3.10,+4.38]` |
| free means / 3 mm cap | **+18.98 dB** | `[+17.13,+20.83]` |

hard 内部的 density×extent interaction 为 **+3.30 dB** `[+2.07,+4.20]`，5/5 为正：同样增加 surface points，只有在 splat 能大幅伸展时，hard RGB 才获得明显额外收益。这是 **cardinality-amplified covariance bypass**。

几何端点给出相反图景。hard unsupported center error 从 0.25× 到 4× 只改善 **−0.42 mm** `[−0.55,−0.28]`；4× 后绝对误差仍为 **35.84 mm** `[31.17,39.97]`，free 则为 **5.20 mm** `[4.84,5.57]`。因此提高 `G_density` 能补采样和增加绘制容量，但 FLAME support surface 本身不包含的头发、颈肩、饰品等区域仍形成稳定中心上限。

三阶 interaction `(mean×extent at 4×) − (mean×extent at 0.25×)` 在 unsupported 为 **+15.18 dB** `[+13.36,+17.05]`，5/5 为正。它说明任何“更密模板已解决几何”的 RGB 结论都必须与 center metric 联合解释。

## F12：跨身份 `N_id` scaling 首次显示 hard support plateau，但 crossover 依赖区域和指标

建立了一个参数匹配的跨身份 NeRSemble 静态多视图代理任务。415 个可用 FREE 身份经同一 landmark-anchored FLAME 流程缓存；训练身份使用嵌套 `N_id={8,32,128,384}`，20 个官方 SVFR 身份从所有训练层完全 held out。每个模型从 `N_obs={1,4,8}` 个输入相机预测其余相机；目标相机不进入 encoder 或 attribute sampling。`hard/adaptive/released G_mu` 使用同一 5,023 slots、固定 5 mm isotropic covariance、hidden 128、总参数 651,816、20,000 steps 和三个训练 seed。36/36 runs 均完成，每 run 有 4,480 个测试区域行；camera 先在身份内平均，seed 再在身份内平均，CI 对 20 个 held-out identity 做 10,000 次 bootstrap。

在预注册的 broad region、`N_obs=4` 下，released 相对 hard 的 PSNR 优势为：

| region | `N_id=8` | `N_id=384` | advantage interaction `384−8` |
|---|---:|---:|---:|
| full frame | **+1.09 dB** `[+0.23,+1.84]` | **+3.37 dB** `[+2.63,+4.08]` | **+2.28 dB** `[+1.85,+2.71]` |
| broad FLAME-unsupported | **+2.85 dB** `[+2.22,+3.51]` | **+6.78 dB** `[+5.57,+7.95]` | **+3.93 dB** `[+2.89,+4.90]` |
| oral | **−0.71 dB** `[−1.08,−0.36]` | **−0.65 dB** `[−0.91,−0.38]` | +0.06 dB `[−0.22,+0.38]` |

hard full-frame PSNR 从 `N_id=8` 的 11.25 dB 只增至 `N_id=384` 的 11.37 dB，而 released 从 12.33 增至 14.73 dB。三个 seed 的 released-hard full-frame effect 在 `N_id=8` 分别为 `+1.23/+1.14/+0.89 dB`，在 `N_id=384` 为 `+3.41/+3.39/+3.31 dB`，不是单 seed 偶然。排除 cache audit 标记的唯一 test identity 133 后，full-frame effect 仍为 `+1.08/+3.40 dB`，方向不变。

但 qualitative audit 发现 broad unsupported 同时含大量 torso、hood 和 shoulder，故保留原结果不动，再用 frozen checkpoint 做 post-audit head decomposition：head ROI 是 face/oral/eyes/scalp/ears 的投影 convex hull，排除 neck，在 128×88 上扩张 6 px。平均每个测试视图约有 4,334 个 head-foreground pixels，其中 3,871 supported、463 unsupported；另有 1,812 torso-outside-head pixels。

head decomposition 的 released-hard PSNR：

| region | `N_id=8` | `N_id=32` | `N_id=384` |
|---|---:|---:|---:|
| head foreground | −0.09 dB `[−0.63,+0.42]` | **+1.78 dB** `[+0.84,+2.73]` | **+2.46 dB** `[+1.67,+3.25]` |
| head supported | **−2.37 dB** `[−2.97,−1.84]` | **−1.56 dB** `[−2.05,−1.09]` | **−0.89 dB** `[−1.27,−0.53]` |
| head unsupported | **+1.51 dB** `[+0.84,+2.23]` | **+4.47 dB** `[+3.08,+5.84]` | **+5.12 dB** `[+3.93,+6.39]` |
| torso outside head | **+2.89 dB** `[+2.02,+3.75]` | **+5.99 dB** `[+4.99,+6.88]` | **+7.09 dB** `[+5.64,+8.44]` |

因此 head-foreground PSNR 的观察符号在 8 与 32 identities 之间改变，但 `N_id=8` 是统计 tie，不是显著 hard win；head-unsupported 的 release 优势从最小规模已为正且继续扩大，是 left-censored crossover；head-supported 与 oral 到 384 identities 仍由 hard 胜出，没有 crossover。torso 放大了 broad effect，但不是 head-unsupported gain 的唯一来源。

更关键的反例来自 LPIPS。正值定义为 release 更好时，released-hard 的 head-foreground LPIPS advantage 在 `N_id=8/384` 分别为 **−0.052/−0.057**，head-supported 为 **−0.052/−0.060**，head-unsupported 为 **−0.051/−0.041**；hard 在所有 head regions 与 identity scales 都保持更好的 spatial LPIPS。只有 torso 从 `N_id=8` 的 −0.004（CI 跨零）转为 `N_id=384` 的 **+0.049** `[+0.023,+0.075]`。

这把“FLAME 是捷径还是上限”改写为更准确的机制结论：**固定 slot budget 下，hard FLAME 是区域耦合的 allocation prior。它把有限 points 集中到模板支撑面，带来 sharp supported detail 和小数据稳定性；release 能随身份规模学习把 slots transport 到模板外头发、轮廓和颈肩，但会稀释 supported density，并在当前训练目标下损害 oral/perceptual detail。所谓 crossover 不是一个全局数字，而是 `data × target region × metric × representation channel` 的 phase diagram。**

该实验仍是静态单帧、低分辨率、固定 covariance/density 的机制代理，不等价于完整 avatar benchmark。由于训练 objective 包含 torso，已在看 follow-up 结果前冻结 head-only target control；capacity、3 mm covariance、60k convergence 和 release-dose controls 已完成，结果见 F13。

## F13：capacity 与 convergence 不会消除 ceiling；covariance 会移动边界；r=30 APR 形成局部 Pareto knee

主实验解盲后冻结四类 follow-up，共 75 runs：3 mm covariance 18 runs、hidden 64/512 各 18 runs、adaptive radius 30/75 mm 各 6 runs、`N_id=384` 的 60k convergence 9 runs。全部通过矩阵、参数、有限值和 source-checkpoint SHA 审计，并用 frozen checkpoints 重评 head ROI。以下均为 `N_obs=4`、20 held-out identities、identity-paired 10,000-bootstrap。

在 `N_id=384` 下，released-hard gap 随控制变化：

| control | head foreground PSNR | head supported PSNR | head unsupported PSNR | head foreground LPIPS advantage |
|---|---:|---:|---:|---:|
| primary：5 mm / h128 / 20k | **+2.46 dB** | **−0.89 dB** | **+5.12 dB** | **−0.057** |
| covariance 3 mm | **+3.33 dB** | **+1.32 dB** | **+5.63 dB** | **+0.029** |
| capacity h64 | **+2.32 dB** | **−0.97 dB** | **+4.95 dB** | **−0.066** |
| capacity h512 | **+2.79 dB** | **−0.68 dB** | **+5.60 dB** | **−0.040** |
| convergence 60k | **+3.04 dB** | −0.42 dB `[−0.93,+0.05]` | **+5.78 dB** | **−0.014** |

capacity 从 615,464 参数的 h64 到 1,213,992 参数的 h512 没有改变区域排序，只将 released advantage 平滑增减约几十分贝。相对 primary，h512 的 released gap 在 head foreground 增加 **+0.33 dB** `[+0.15,+0.53]`，head unsupported 增加 **+0.48 dB** `[+0.16,+0.85]`；h64 分别减少 −0.14 与 −0.17 dB。因而主曲线不是特定 decoder 宽度的偶然。

60k 更直接否定“20k 欠拟合制造 ceiling”。从 20k 到 validation-selected 60k：

- hard head-foreground 只由 13.91 增至 13.94 dB，released 由 16.37 增至 16.98 dB；
- hard head-unsupported 由 5.688 变为 5.690 dB，released 由 10.81 增至 11.47 dB；
- released-hard gap 相对 primary 在 head foreground 增加 **+0.58 dB** `[+0.28,+0.90]`，unsupported 增加 **+0.66 dB** `[+0.27,+1.10]`；
- head-supported PSNR 与 oral PSNR 的 gap 均收缩到 CI 跨零，但 head LPIPS 仍小幅偏好 hard。

covariance 则是强交互项。3 mm 下 released 在 `N_id=384` 的 head-supported PSNR 从 primary 的 −0.89 dB 翻为 **+1.32 dB** `[+0.68,+1.95]`，head-foreground LPIPS advantage 从 −0.057 翻为 **+0.029** `[+0.013,+0.045]`。但 3 mm 的绝对 LPIPS 明显更差：head foreground hard/released 为 0.432/0.403，而 5 mm 为 0.231/0.288。即 relative reversal 是 narrow splats 令 hard surface sampling 更早崩溃，不是 3 mm 系统整体更优。这与 static oracle 的 covariance bypass 一致：**`G_Sigma` 会改变测得的 `G_mu` crossover，甚至改变指标排序。**

release-dose 给出 APR 的首个 population-level 正结果，但边界很窄。`N_id=384` 时，adaptive r=30 的平均 displacement 仅 7.62 mm，released r=150 为 32.34 mm。r=30 相对 hard：

| head region | PSNR effect | identity-bootstrap 95% CI |
|---|---:|---:|
| foreground | **+1.56 dB** | `[+1.03,+2.09]` |
| supported | −0.14 dB | `[−0.34,+0.04]` |
| unsupported | **+2.31 dB** | `[+1.60,+3.00]` |

按每个 identity 的实际 displacement，在 hard 与 released 端点之间线性插值质量，r=30 位于 chord 之上的 residual 为 head foreground **+0.99 dB** `[+0.60,+1.38]`、head unsupported **+1.12 dB** `[+0.51,+1.69]`，supported +0.10 dB `[−0.10,+0.28]`。这支持“用约四分之一 displacement 获得超线性 PSNR 收益”的 APR knee。

但 r=30 对 hard 的 head-foreground/head-supported LPIPS 仍分别差 −0.012/−0.014；它相对 released 则好 **+0.045/+0.046**。所以 H3 的准确结论是：**简单 evidence gate 在大 identity scale 上改善 quality–displacement 与 coverage–sharpness Pareto 前沿，但没有跨 PSNR/LPIPS 全指标支配 hard。** `N_id=8` 时 r=30 的 head-foreground PSNR 为 −0.22 dB `[−0.61,+0.19]`，也不存在小数据收益；adaptive release 的 knee 本身随数据规模出现。

## F14：target region 是 allocation 的因果开关；2× density 只部分修复但不移除 support ceiling

qualitative audit 发现 primary full-foreground objective 会奖励 torso/clothing 后，在看结果前冻结两个 post-primary mechanism controls：

1. `head-target`：只在 FLAME head envelope 内计算 RGB/alpha loss，背景泄漏只约束局部外环；模型、5,023 slots、5 mm covariance、20k steps 和 splits 不变；
2. `density2`：full target 不变，保留全部 5,023 vertex slots，再加 5,023 个跨身份共享 face/barycentric slots；新增 slots 接回三个父顶点做平滑。三方法在 2× 内参数匹配，总参数由 651,816 增至 812,552。

两套各 18 runs，`N_id={8,384} × 3 methods × 3 seeds`。全部 ordinary/head-ROI source SHA、4,480/2,560 行与有限值审计通过。

### Head-only target 直接重排 query allocation

`N_id=384, N_obs=4` 时，head-target 相对 full-target 的 method-level change：

| method | head foreground PSNR | head supported PSNR | head unsupported PSNR | torso PSNR |
|---|---:|---:|---:|---:|
| hard | −0.01 dB | −0.00 dB | −0.00 dB | −0.08 dB |
| released | **+0.87 dB** `[+0.45,+1.27]` | **+1.46 dB** `[+1.13,+1.81]` | −0.21 dB `[−1.15,+0.64]` | **−7.36 dB** `[−8.73,−5.92]` |

hard 几乎完全不受 target switch 影响，因为固定 surface slots 本来就不能重分配。released 则把原来用于 torso 的能力明确换回 head；其 head-foreground LPIPS 改善 **+0.054** `[+0.043,+0.065]`，hard 只变化 +0.002。

因此 released-hard regional gap 从 primary 变为：

| scale/region | primary full target | head-only target |
|---|---:|---:|
| `N_id=8`, head foreground PSNR | −0.09 dB（tie） | **+0.57 dB** `[+0.23,+0.94]` |
| `N_id=8`, head supported PSNR | **−2.37 dB** | **−0.85 dB** `[−1.46,−0.34]` |
| `N_id=8`, head unsupported PSNR | **+1.51 dB** | **+1.21 dB** `[+0.74,+1.75]` |
| `N_id=384`, head foreground PSNR | **+2.46 dB** | **+3.34 dB** `[+2.57,+4.16]` |
| `N_id=384`, head supported PSNR | **−0.89 dB** | **+0.57 dB** `[+0.36,+0.76]` |
| `N_id=384`, head unsupported PSNR | **+5.12 dB** | **+4.92 dB** `[+3.92,+5.97]` |
| `N_id=384`, torso PSNR | **+7.09 dB** | −0.19 dB `[−0.38,+0.02]` |

head-target 下 `N_id=384` 的 head-foreground/supported/unsupported 三个 released effect 在三个 seeds 分别为 `+3.324/+3.313/+3.373`、`+0.617/+0.549/+0.555`、`+4.862/+4.873/+5.019 dB`。head LPIPS 的 released-hard gap 在所有三个 head regions 均缩到 CI 跨零，不再由 hard 显著胜出。

几何端给出机制证据。`N_id=384` released 的平均 displacement 从 32.34 降至 14.23 mm；neck 从 101.22 降至 **19.86 mm**，scalp 从 55.21 降至 20.77 mm，face 从 22.03 降至 14.13 mm。neck change 为 **−81.37 mm** `[−93.36,−68.13]`。这不是模糊的 regularization 叙事，而是 target objective 直接决定 finite slots 被 transport 到哪里。

结论：primary 的 supported/LPIPS penalty 主要来自 **全 foreground 目标诱导的区域 allocation conflict**，不是 release 的必然代价；但 head-unsupported 的约 +5 dB gain 在移除 torso 奖励后仍保留，故也不是 torso artifact。target region 本身必须进入 phase diagram。

### 2× surface density 部分缓解分配冲突，但 hard support ceiling 保留

`N_id=384` 时，2× density 相对 1×：

- hard/released head-supported PSNR 分别提高 **+0.49/+0.91 dB**，released-hard supported gap 从 −0.89 收缩为 **−0.48 dB** `[−0.75,−0.25]`；gap change 为 **+0.41 dB** `[+0.12,+0.74]`；
- head-foreground released-hard gap为 **+2.63 dB** `[+1.84,+3.41]`，与 1× 的 +2.46 接近；
- head-unsupported released-hard gap仍为 **+4.79 dB** `[+3.74,+5.92]`，三个 seeds 为 `+4.642/+4.987/+4.748 dB`；
- 2× 对 hard head-unsupported 只提高 +0.24 dB，对 released 为 −0.09 dB；release gap略收缩 −0.33 dB `[−0.60,−0.06]`，但远未消失；
- head-foreground LPIPS gap从 −0.057 收缩为 **−0.027** `[−0.037,−0.017]`，仍偏好 hard。

在 `N_id=8`，2× 并未把 crossover 向左移动：head-foreground released-hard 仍为 −0.27 dB `[−0.80,+0.24]`，head-unsupported gain反而从 +1.51 降至 **+0.91 dB** `[+0.40,+1.45]`。因此预注册的“更多 slots 会让 released 在小数据更早胜出”被否定。

2× 后 adaptive/released 的平均 displacement 分别从 26.65/32.34 降至 20.11/23.74 mm，说明增加 surface samples 改变了 transport demand；但额外 points 仍位于同一 FLAME support，不能直接表示模板外区域。**density 可以减轻 sampling 与 allocation 压力，却不能替代 support release。**

## 当前能说与不能说

可以说：

- `G` 必须拆成 `G_mu`、`G_Sigma` 和 `G_density`；三者的五案例因子交互已经实际测量；
- tight FLAME mean support 在 static oracle 中形成强 center-geometry ceiling；
- 大 covariance 会显著改变 hard/free RGB gap，而且可在中心错误不变时提高图像质量；
- release dose 存在宽 knee，单一 hard/free 端点遗漏结构。
- held-out view 上 bounded release 可优于 hard/free 的 supported 区域，而 unsupported 区域继续偏好 free；区域级 release 有直接实证动机。
- 在五身份动态 mouth protocol 中，8 帧时 hard A 一致优于 free A，而 full observation scale 时符号反转；adaptive A 在 oral 区域形成更优端点；
- A 的最优 release capacity 随 observation scale 改变，过大容量在小数据下显著扩大泛化 gap；
- C 的 attribute identity、A 的 deformation mapping 和 P 的 tracking timing 已经通过中心不变/角色固定的受控干预分离；
- P lag 对高精度 adaptive A 的伤害可大于对 hard A 的伤害，P correction 不能由 representation release 替代。
- 增加 FLAME-surface density 不能消除 unsupported center ceiling，却会与 wide covariance 联合抬高 hard RGB；更高表示容量可能让错误几何更难从图像指标中发现。
- 跨身份训练中，hard `G_mu` 的 broad/head-unsupported PSNR 出现近乎不随 `N_id` 改善的 support plateau，而 released gain 随 `N_id` 显著扩大；该结论在三个 seed 和 scaffold sensitivity 中方向一致。
- 同一个模型不存在全局 FLAME crossover：head-foreground PSNR 在 8–32 identities 间出现观察符号变化，head-supported/oral 始终偏好 hard，head-unsupported 始终偏好 release且优势随规模扩大。
- PSNR 与 LPIPS 给出相反的 head 结论，说明当前 fixed-density release 是 coverage–sharpness allocation frontier，不能把单一 PSNR 提升写成全面质量胜利。
- decoder capacity 64–512 与 20k→60k convergence controls 不会消除 hard head-unsupported plateau；增加优化预算反而扩大 released advantage。
- `G_Sigma=3/5 mm` 能改变 `G_mu` 的 supported-region 与 LPIPS 排序，FLAME prior scaling 必须报告联合 `G_mu × G_Sigma` phase boundary。
- adaptive r=30 在 `N_id=384` 以约四分之一 released displacement 获得高于 hard↔released chord 的 head PSNR，形成 population-level APR Pareto knee；但它仍不支配 hard LPIPS。
- head-only target 把 released 的 torso gain归零、将 neck displacement 降约 81 mm，同时保留约 +4.9 dB head-unsupported gain；target-region allocation 是独立因果轴。
- 2× FLAME-surface density 缩小 supported/LPIPS penalty，但不移除约 +4.8 dB head-unsupported release advantage；sampling density 不是 support ceiling 的替身。

暂时不能说：

- 已建立适用于完整 avatar 系统的普遍 `N_id` scaling law；当前跨身份证据是静态单帧、低分辨率、固定 covariance/density 的机制代理；
- 已建立动态 novel-view 结论；当前官方动态公开训练数据只有一个 RGB camera；
- 舌头动画已解决；动态公开 benchmark 不含 tongue sequence，伸舌证据仍是静态多视图；
- 自然 tracking error 的真实分布与影响已测得；当前 P 是 controlled lag；
- 完整 APR 已优于所有固定 residual/free；当前简单 `G_mu` gate 与 A spatial gate 尚未证明跨区域、跨指标的 Pareto 支配，更未联合 `G/C/P` evidence；
- adaptive r=30 是普遍最优 release；当前正结果只在大 identity scale 的 quality–displacement frontier 上成立，小数据与 LPIPS 仍有反例；
- “小数据 hard FLAME 全局更优”；head-only target 下 released 在 `N_id=8` 的 head-foreground PSNR 已显著为正，但 supported detail 仍偏好 hard。H1 必须限定 region、metric 与 target objective；
- C shuffle 证明任何无 FLAME 方法都失败；它只识别 shared-attribute renderer 对固定身份的依赖；
- 当前已经达到 SIGGRAPH 主会证据量。

## 下一组决定性实验

1. 用完整 multi-camera 原始动态数据或可公开替代集复现 A crossover，补 novel-view/geometry endpoint。
2. 把静态 `N_id × target × density` 发现迁移到动态、多帧、多视图 avatar benchmark，验证 phase boundary 是否保留。
3. 将 `C-local soft transport` 与 `P-confidence correction` 做成可学习分支，而不是 controlled corruption。
4. 把已验证的 A gate 扩展为 evidence-conditioned `rho_G/rho_C/rho_A/rho_P`，比较 fixed residual、global gate、scheduled release、parameter-matched free。
5. 对 mouth localization 反例做独立标注/光流分析，确认是序列联动还是 gate 机制失败。

## 可复查产物

- 伸舌 release curve：`artifacts/analysis/tongue017_frame182_landmarkanchored4000/`
- 伸舌 mean×extent：`artifacts/analysis/tongue017_frame182_scale_caps4000/`
- 伸舌四折 held-out release：`artifacts/analysis/tongue017_frame182_release_cv4000/`
- 五案例 mean×extent：`artifacts/analysis/multicase_landmark_mean_extent4000/`
- 五案例 mean×extent×density：`artifacts/analysis/multicase_mean_extent_density4000/`
- 五案例三折 held-out release：`artifacts/analysis/multicase_release_cv4000/`
- 五身份 A observation-scale crossover：`artifacts/analysis/animation_multisubject_exp5/`
- A capacity × scale：`artifacts/analysis/animation_capacity_exp5/`
- 四 motion family 区域交互：`artifacts/analysis/animation_multisequence/`
- P tracking lag：`artifacts/analysis/tracking_lag/`
- C correspondence intervention：`artifacts/analysis/correspondence/`
- 跨身份主实验：`artifacts/analysis/identity_scaling20k_v1/`
- 跨身份 head-vs-torso decomposition：`artifacts/analysis/identity_scaling20k_v1_head_roi/`
- 跨身份 dataset/cache audit：`artifacts/analysis/nersemble_scale_cache/`
- 跨身份 covariance/capacity/convergence/dose controls：`artifacts/analysis/identity_scaling_followups_v1/`
- 跨身份 target-region 与 2× density controls：`artifacts/analysis/identity_scaling_mechanism_controls_v1/`
- 可信 FLAME overlays：`artifacts/cache/flame_fit_*_landmark_anchored_overlay.png`
