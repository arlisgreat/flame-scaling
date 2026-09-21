# 实验协议

## Stage 0：工程与测量校准

目标是先消除会让所有后续结论失效的实现问题。

1. 重构共同 renderer，使 `G/C/A/P` 可单独关闭；测试任何 `free` 分支都不会在后续层重新读取 FLAME。
2. 在一个 subject、一个 frame、所有相机上过拟合，验证 camera convention、mask、alpha、色彩和 metric。
3. 使用 GT point cloud / mesh 做 render-evaluate positive control；稀疏点云低分不能当算法结论。
4. 做 2k-step A800 profiling，再冻结每个 scale 的 token、显存、吞吐和预计 wall time。
5. 固定 split、区域 mask、yaw/expression bins 和失败样本处理规则。

退出条件：三次独立初始化均可稳定过拟合；held-out view 不是空白/错裁剪；硬/free 只在预期代码路径不同。

## Stage 1：Oracle representation ceiling

这是 H2 的必要前置实验。对同一真实多视图 frame/sequence，在使用全部训练视图与可靠 calibration 的条件下做 per-instance optimization，最大限度移除泛化与 tracker 混淆。

### 方法 ladder

- surface-hard：严格贴 FLAME surface，固定 `C/A`；
- bounded-offset：偏移半径 `{2, 5, 10, 25} mm`；
- unbounded-residual：同 slot、同 `C/A`，只解除坐标边界；
- free-geometry：保持 slot ID，坐标独立学习；
- residual-animation：FLAME deformation + SE(3) residual；
- free-animation：共享 `z_dyn` 驱动 learned deformation；
- fully-free：移除 FLAME `G/C/A`，但 Gaussian 数、SH、optimizer 和 steps 相同。

上述每个 `G_mu` rung 必须声明并控制 `G_Sigma` 与 `G_density`。主 oracle 至少加入 `mean={hard,bounded,free} × axis-cap={tight,wide} × Gaussian-count={matched}`；否则 stretched splats 会把错误中心“画对”，使 RGB ceiling 被低估。主表同时报告 mean-to-GT geometry、scale quantile、cap saturation 和 held-out-view 指标。

### 数据与区域

- NeRSemble：舌头 1/2、eyes、lips、mouth、jaw、cheeks+nose、emotion、speech；
- FaceScape / RenderMe-360：如许可与本地数据允许，用于细节 mesh/全头区域复核；
- 合成 stress set：已知 tongue/hair/accessory deformation 和真值 control，用于 `P-oracle`。

预注册区域：稳定皮肤、眼周、嘴唇、牙齿、舌头、动态皱纹、耳朵、头发/胡须、饰品、颈肩、背景。主结论不得只使用全脸平均。

### ceiling 证书

某区域被称为 representation bottleneck，至少满足：

1. hard 与 relaxed/free 在三种初始化、加长优化和同等表示预算下仍有显著 oracle gap；
2. gap 同时体现在图像与至少一种 geometry/跨视图指标；
3. `P-oracle/selfcal` 下 gap 仍存在；
4. 增加 hard 模型容量或优化预算不再有效，而 release 有效。

若 hard 只在 train-view RGB 上追平、center geometry 仍错或 held-out view 下降，应报告为 covariance shortcut，而不是解除 representation bottleneck。

## Stage 2：Scaling curves

将身份多样性和每身份观测密度分开，避免把“更多身份”和“更多视图/帧”混成数据规模。

### 主轴

- 身份数 `N_id`: `{32, 128, 512, 2048}`；若完成 identity 去重，再扩展到约 `8k–15k` 视频/身份代理规模；
- 每身份观测 `N_obs`: `{1, 4, 16, 64}`，静态视图与时序帧分开实验；
- decoder 容量：`S/M/L`，目标约 `{45M, 180M, 650M}` 总参数，但最终以实测 parameter matching 为准；
- 方法：`hard`、`free`、`APR`；
- seed：`{0,1,2}`，split 与采样顺序配对。

主实验先固定 `N_obs=16`，得到 `4 data × 3 capacity × 3 method × 3 seed = 108` runs。观测密度单独在 `N_id={128,2048}` 上扫描，防止主矩阵爆炸。

### 数据分工

- 公共大规模 2D/视频数据：训练 sample-efficiency 与身份 scaling，先做身份去重和许可审计；
- NeRSemble 418 subjects：公共多视图 stress validation，特别是舌、口、眼、jaw 与跨视图；
- RenderMe-360 / FaceScape：仅在许可允许时作外部复核；
- 私有百万身份结果只作为相关工作参照，不能成为本论文关键证据。

训练集 scale 必须嵌套：`D_32 ⊂ D_128 ⊂ D_512 ⊂ D_2048`。测试 identity、sequence、camera 全部冻结。

### 公共数据静态 proxy（结果揭盲前冻结于 2026-07-15）

在获得更大且许可清晰的数据前，先用完整 NeRSemble v2 做一个保守的跨身份 `G_mu` 曲线。该实验不替代上面的最终动态主矩阵，作用是检验 418 身份范围内是否已经出现方向性 crossover，并为后续算力分配提供依据。

- 数据：`FREE` 每身份一个静态 frame；由固定 source camera 的 7 个等距候选帧中选择 oral aperture 最大者；
- split：391 个非 benchmark 可用身份按固定 hash 排序，嵌套训练集 `N_id={8,32,128,384}`；余下 7 个固定为 validation；20 个官方 SVFR 身份固定为 test；
- camera：415 个可用身份共同存在的 15 个 camera；`N_obs={1,4,8}` 是从 `222200037` 开始的确定性最远角度嵌套前缀，所有 target camera 必须在前缀之外；
- scaffold：每身份用全部可用视图的 68 点三角化拟合 FLAME，故是对 hard 极有利的 multiview-oracle initialization；它只允许建立“即使给 hard oracle scaffold 仍如何”的保守结论；
- 输入：只读取观测 RGB/alpha、camera 和 fitted scaffold；禁止读取由全部视图投影得到的 cache `initial_colors`；
- 共享模型：相同 observation encoder、FLAME slot sampler、attribute decoder、5023 semantic slots 和总参数；
- 唯一干预：`hard` 为 0 位移；`adaptive` 为带稀疏 evidence-conditioned gate 的 150 mm 最大有界位移；`released` 为无 gate 的 150 mm 有界位移。adaptive 与 released 共享最大支撑，以免把 gate 收益混成 radius 收益；三者固定 5 mm 各向同性 covariance，禁止 `G_Sigma/G_density` 补偿。5 mm 是在揭盲跨身份结果前以单身份 surface-coverage positive control 冻结的最小连续覆盖值；3 mm 作为端点敏感性控制。后续另做 30/75 mm dose control；
- 合理性约束：所有非 hard 位移共享 FLAME mesh edge 上的 normalized offset Laplacian（weight 2.0）和实际位移幅度正则（0.01）；adaptive 另有 gate L1（0.003）与 gate Laplacian（0.1）。这些权重在跨身份主结果揭盲前由 0.05/0.5/2.0 连续几何 smoke 冻结，2.0 首次消除明显独立点 jitter；目的是禁止不合理噪声冒充有效 release。它保留本实验固定的 `C`，不额外引入 target 几何；
- 优化：3 个配对训练 seed `{0,1,2}`，完整 proxy 为 `4 N_id × 3 method × 3 seed = 36` runs；每个 20k equal-step，AdamW + cosine decay 到初始学习率的 0.05 倍，validation foreground MAE 选 checkpoint；test 不参与选择；
- 报告：PSNR/MAE/SSIM 与 AlexNet-LPIPS spatial map 的区域均值；target camera 先在身份内聚合，再对 20 个身份做 cluster bootstrap 10k 次；区域为 full frame、foreground、FLAME-supported、FLAME-unsupported、oral、eyes、background；
- 解释边界：`released` 仍保留 FLAME initialization 与 `C`，不是 fully no-FLAME；单静态帧不能证明 `A/P` 或动态容量结论。

代码合同为 `scripts/audit_nersemble_identity_scaling.py`、`scripts/prepare_nersemble_scale_subject.py`、`scripts/run_identity_scaling.py` 和 `scripts/analyze_identity_scaling.py`。协议在完整矩阵结果产生前写入，后续任何修改都必须作为新版本并说明原因。

## Stage 3：角色消融与交互

在低/中/高三个 anchor 上执行：

- 单独释放 `G`、`C`、`A`、`P`；
- `G×C`：几何与语义 ordering 是否互相补偿；
- `G×A`：geometry ceiling 是否被 deformation residual 缓解；
- `A×P`：动画表示与 tracker 污染是否混淆；
- release dose：offset radius、transport entropy、residual motion norm、tracker correction strength。

使用混合效应模型估计 main effect 与预注册 interaction，不对所有组合做事后挑选。

## Stage 4：APR 与 Pareto

比较 fixed-hard、fixed-free、global schedule、global gate、region gate、full APR。总参数、Gaussian 数、输入、训练 token 和 augmentation 相同。

Pareto 维度：

- 区域质量与跨视图一致性；
- 几何误差；
- 新身份所需样本数；
- 训练 FLOPs / inference latency / 显存；
- 动画控制误差与时间稳定性。

“APR 最优”要求在 bootstrap 不确定性下仍为非支配解；单一 LPIPS 提升不成立。

## 指标

### 主要终点

- region-masked LPIPS / DISTS；
- calibrated novel-view depth/normal 或 point-to-surface；
- multiview reprojection / feature consistency；
- tongue+oral、hair、stable-skin 三个预注册区域组的 crossover。

### 次要终点

- PSNR、SSIM；
- identity similarity（独立识别器）；
- landmark/AU、optical-flow motion error；
- temporal LPIPS / flicker；
- yaw bins：`0–30°`, `30–60°`, `>60°`；
- expression bins 和 sequence family；
- total/active parameters、FLOPs、train tokens、latency。

FLAME tracker 导出的 AED/APD 只能做次要指标，不能用它证明 FLAME 方法更准。

## 算力计划

本机有 8×A800 80GB。先 profiling，再承诺总预算：

- pilot：24–36 runs，验证因果开关、metric 和三条趋势；
- oracle：按 region/sequence 分片，多初始化；
- main 108 runs：8 GPU 并行，保存统一 ledger；
- role/interaction/APR：只有 pilot 达到继续条件后启动。

完整研究预计是数百 A800-GPU-days 量级；在没有吞吐 profiling 前不写虚假的精确周数。每阶段设置 compute kill gate，防止在测量管线错误时消耗大规模算力。

## 继续 / 停止条件

- Stage 0 失败：停止训练，修 renderer/camera/metric；
- oracle 无 gap：不宣称该区域有 representation ceiling，转而报告 negative finding；
- scaling 无 crossover：报告右删失阈值或 hard/free 一直占优，不外推；
- 差距被 `P-selfcal` 消除：结论改为 supervision bottleneck；
- APR 被 parameter-matched fixed residual 追平：删除 APR 方法贡献，保留 causal study；
- 结果仅在私有/不可发布数据成立：不提交主结论。
