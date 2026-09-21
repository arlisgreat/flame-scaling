# FLAME: Shortcut or Ceiling?

这是一个面向可证伪结论的研究仓库，研究硬 FLAME 偏置何时是 `sample-efficient prior`，何时成为 `representation bottleneck`。

核心不是做一次 `FLAME vs. no-FLAME` 总分对比，而是分离 FLAME 的四个作用：

- `G`：几何初始化与几何约束；
- `C`：FLAME 语义对应、固定 query 身份和邻域；
- `A`：LBS、blendshape 与动画控制空间；
- `P`：tracker 产生的姿态、表情和相机伪标签。

当前实验又揭示了一个必须先拆开的内部混淆：在 3DGS 中，`G` 不是单一开关。至少要区分 Gaussian mean 支撑 `G_mu`、covariance/extent `G_Sigma` 与 cardinality/densification `G_density`。固定 FLAME surface 但允许大 Gaussian 或更多 splats，仍可绕过部分 RGB 上限，而不修复中心几何。

研究结论必须同时回答三个问题：

1. 在相同表示预算下，硬约束是否存在无法由更多数据消除的 oracle-fit 误差？
2. 随身份数、每身份观测量和模型容量增加，每个区域、每个 FLAME 作用的收益曲线在哪里交叉？
3. 自适应释放是否在质量、数据、算力、可控性组成的 Pareto 前沿上优于硬先验和固定自由模型？

## 当前判断

问题真实、可发表，但原始命题过宽。HeadsUp 已研究身份数、视图数和容量 scaling；NPGA、OMEGA-Avatar、FFAvatar 等已包含不同形式的 residual、template relaxation 或局部 offset。够强的新贡献应是：

- 首个四角色因果拆分；
- 区域级 prior phase diagram 与有置信区间的 crossover；
- 用 oracle-fit 将表示上限与数据不足、tracker 噪声分开；
- 基于证据而非统一日程的 Adaptive Prior Release（APR）；
- 可复现的公共数据划分、区域标注和统计协议。

详细判断见 [研究主张](docs/00_verdict_and_thesis.md)、[相关工作证据](docs/01_literature_evidence.md)、[因果拆分](docs/02_causal_factorization.md)、[实验协议](docs/03_experiment_protocol.md)、[当前发现](docs/07_current_findings.md)、[英文论文工作稿](docs/08_paper_draft.md) 和 [可复现 ledger](docs/09_reproducibility_ledger.md)。

## 当前实证进度

- 真人伸舌帧的 mean-release ladder 显示 unsupported 区域从 hard 的 11.74 dB 提升到 free 的 36.25 dB，但直接 oral-protrusion ROI 只提升 1.15 dB。
- `mean constraint × Gaussian extent` 因子实验解释了该反直觉结果：3 mm extent 下 oral free-hard 差为 +4.67 dB，30 mm 下缩至 +1.15 dB；hard 单靠放宽 extent 获得 +5.46 dB。
- 五个真人案例的 2×2 配对复现中，unsupported 区域的 extent-channel 差分中的差分为 +1.79 dB，bootstrap 95% CI `[+0.60,+2.98]`，5/5 案例为正。
- 五案例 `G_mu × G_Sigma × G_density` 的 2×2×3 因子实验显示：0.25×→4× density 时，hard/3 mm 的 unsupported PSNR 只增 +0.52 dB，hard/30 mm 增 +3.82 dB，形成 +3.30 dB density×extent interaction；但 hard 的中心误差仍为 35.84 mm，而 free 为 5.20 mm。更多 anchored splats 会放大 covariance bypass，不能消除错误支撑面。
- 伸舌帧的四折 angular held-out curve 进一步显示区域最优 release 不同：oral 由 75 mm 最优，而 broader unsupported 由 free 最优。五 subject 三折复现中，75 mm 在 supported 区域比 free 高 +0.13 dB `[+0.01,+0.24]`，在 unsupported 区域反而低 −0.45 dB `[-0.89,−0.00]`。
- 动态 `A` 在五身份 mouth sequence 上出现 observation-scale crossover：8 帧时 free−hard oral 为 −4.31 dB `[−5.45,−3.37]`，full sequence 时反转为 +3.69 dB `[+2.83,+4.73]`；A-adaptive 在 full oral 比 hard 高 +5.26 dB、比 free 高 +1.57 dB。
- `data × capacity` 交互不是单调 scaling：8 帧 free-A 随 K=4→64 恶化 6.01 dB；full-scale adaptive 在 K=32 最优。whole-head 偏好 free，oral/eyes 偏好 adaptive，形成动态区域 phase diagram。
- 四 motion family 的 target-control 检验中，head、eyes、jaw 的 motion-aligned interaction 为正；mouth localization 区间跨零，是保留的负结果。
- `P` 的受控 tracking lag 产生单调污染；更反直觉的是 adaptive A 相对自身 clean baseline 比 hard 更敏感，说明 representation release 不能替代 pseudo-label correction。
- `C` 的中心不变干预表明，逐帧打乱 Gaussian attribute identity 会产生 dose-dependent loss；10 mm 邻域 local correspondence 明显优于 global correspondence，但更多数据仍不能清零损失。
- 旧的弱正则 point-cloud FLAME scaffold 结果已降级；当前五案例全部使用多视图稳定关键点锚定、系数限制在 ±3 的 scaffold。
- 跨身份 `N_id × N_obs` 静态 proxy 已完成：36 个 primary runs，391 个非 benchmark 身份提供嵌套 `8/32/128/384` 训练集，7 个 validation，20 个官方 SVFR test，15 个共同 camera；RGB target 严格隔离，且不读取全视图生成的 `initial_colors`。hard head-unsupported 基本平台，released-hard gap 从 `N_id=8` 的 +1.51 dB 增至 `N_id=384` 的 +5.12 dB。
- 75 个 covariance/capacity/convergence/dose controls 已完成。hidden 64/128/512 不改变排序；60k 把 unsupported gap 增至 +5.78 dB；3 mm covariance 会改变 supported 与 LPIPS 的相对排序，说明 `G_Sigma` 会移动 `G_mu` phase boundary。
- adaptive r=30 在 `N_id=384` 用约四分之一 released displacement，相对 hard 提高 head foreground +1.56 dB、unsupported +2.31 dB，并高于 hard→released displacement chord；但 LPIPS 仍不支配 hard，是局部 Pareto knee。
- 18 个 head-target controls 表明 full-foreground objective 会诱导 query transport 到 neck/torso。切换为 head-only 后 released neck displacement 从 101.22 降至 19.86 mm，torso advantage 归零，head-supported gap从 −0.89 翻为 +0.57 dB，而 unsupported gain仍为 +4.92 dB。
- 18 个 2× density controls 表明更多 FLAME-surface slots 只部分缓解 supported/LPIPS penalty；`N_id=384` 的 unsupported release advantage仍为 +4.79 dB，support ceiling 没被 10,046 anchored queries 消除。

完整结果与限制见 [当前发现](docs/07_current_findings.md)。当前已建立 static held-out-view `G`、跨身份 `N_id`、单相机 temporal-held-out `C/A/P` 和 target/density mechanism controls；尚未建立动态 multi-view identity scaling、自然 tracker-error 分布或联合 `G/C/A/P` APR。

## 仓库结构

```text
configs/study.json             预注册式实验因子与主终点
docs/                          论点、证据、协议、统计和风险
schemas/result_columns.md      每个 run 必须记录的结果字段
scripts/audit_local_assets.py  只读审计本机数据、代码和 GPU
scripts/generate_run_matrix.py 生成主实验与角色消融矩阵
scripts/analyze_crossover.py   配对差值、bootstrap CI 与交叉区间
scripts/run_animation_oracle.py 动态 A/C/P 受控干预与 temporal held-out 评估
scripts/run_geometry_oracle.py  静态 G_mu/G_Sigma/G_density oracle 与 held-out-view 评估
scripts/run_identity_scaling.py 跨身份 hard/adaptive/released G_mu 缩放实验
scripts/analyze_identity_scaling.py 身份级配对 bootstrap、crossover 与 phase diagram
scripts/analyze_identity_scaling_followups.py covariance/capacity/convergence/dose controls
scripts/analyze_identity_scaling_mechanism_controls.py target-region 与 density controls
artifacts/                     生成的审计和实验表，不存原始人脸数据
```

## 快速开始

```bash
python3 scripts/audit_local_assets.py --output artifacts/audits/local_assets.json
python3 scripts/generate_run_matrix.py --config configs/study.json --output artifacts/run_matrix.csv
python3 scripts/analyze_crossover.py --input artifacts/example_results.csv --metric lpips --lower-is-better
```

`/data1/workspace/airulan/3dv` 仅作为只读数据与已有实现来源；本研究的代码和结论留在本仓库。任何原始 RGB、身份信息、checkpoint 或 tracker 结果都不得提交。
