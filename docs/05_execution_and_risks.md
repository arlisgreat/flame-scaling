# 执行顺序、资源与主要风险

## 已有本地条件

- 8× NVIDIA A800-SXM4-80GB；
- NeRSemble v2 本地约 1.5TB，418 participant 目录，16-camera calibration 与 tongue/eye/lip/mouth/jaw 等序列；
- NeRSemble benchmark 本地约 3.3GB；
- LAM 代码、资产与 checkpoint 本地约 7.3GB。

上述状态由 `scripts/audit_local_assets.py` 复核。原始人脸数据保持只读且不进入本仓库。

## 两个近期硬风险

### 1. 公共大规模训练集

NeRSemble 的 418 identities 足以做多视图 stress evaluation 和中等规模曲线，但不足以独立支撑“继续扩到万级仍发生交叉”的论断。VFHQ/其他公开视频可提供规模，但必须先解决身份去重、许可、帧相关性和姿态分布。HeadsUp/FFAvatar 的关键大数据是私有数据，不能把它们的 scale 当作可复现实验。

决策：首稿把可确认的 `32→418/512` 公共曲线与 oracle ceiling 做扎实；万级 2D/video scaling 是扩展轴，不允许污染主因果结论。

### 2. LAM 不是现成训练基座

公开 LAM 代码适合复用 query/renderer 结构和 checkpoint，但本地审计表明发布版本缺少可直接运行的完整训练 runner。更关键的是，现有 renderer 在部分 learned-query 路径仍重建 FLAME geometry；简单切到 embedding 并不等于 no-FLAME。

决策：抽取最小共享 `CanonicalScaffold → AttributeDecoder → Deformer → GaussianRenderer`，每层显式声明是否读取 FLAME，并写 dependency-trace test。不要在无法训练的推理仓库上堆大量 patch。

## 六周最小证据路径

### Week 1：测量闭环

- 实现最小 renderer adapter 和 `G/C/A/P` dependency test；
- NeRSemble subject 017 的 calibration、mask、region、held-out camera smoke test；
- GT geometry / synthetic primitive positive controls；
- 2k-step profiling。

### Week 2：oracle pilot

- stable skin、mouth、tongue 三类区域；
- hard / 5mm / 25mm / unbounded / free geometry；
- 三个初始化，equal-step + converged；
- 得到第一张 `constraint strength × region error` 图。

### Week 3：P 与 A 去混淆

- fixed tracker vs self-cal/oracle；
- FLAME-only vs shared-latent residual animation；
- 判断看到的是 representation ceiling 还是 supervision/control ceiling。

### Week 4：小规模 learning curve

- `N_id={8,32,128}`，`capacity={S,M}`；
- hard/free/fixed residual；
- 检查是否存在 H1 方向与初步 crossover。

### Week 5：APR pilot

- global gate、region gate、parameter-matched residual；
- gate 与独立 oracle gap 的相关性；
- 通过后才扩 main matrix。

### Week 6：内部审稿

- 冻结预注册协议；
- 复核 novelty threat；
- 计算主实验 GPU-day 与磁盘；
- 以“哪条结论会被哪个实验推翻”为主线重写 paper outline。

## 里程碑门槛

| 里程碑 | 继续条件 | 失败后的动作 |
|---|---|---|
| M0 renderer validity | positive control、camera、region metric 全通过 | 不启动模型训练 |
| M1 oracle ceiling | 至少一个非 FLAME 区域存在稳健 gap | 改为“先验并未构成上限”的负结果 |
| M2 sample efficiency | 小数据至少一个主终点 hard 更优 | 不宣称 shortcut，研究变为 ceiling-only |
| M3 crossover | 观测范围内有 CI 支持的交叉，或清晰删失结论 | 不做外推阈值 |
| M4 APR mechanism | 超过参数匹配控制且 gate 对齐独立证据 | 删除方法贡献，保留 findings paper |

## SIGGRAPH 工作量判定

达到主会量级需要同时具备：

1. 一个难以被替代解释推翻的主要发现；
2. oracle + scaling + causal role ablation 三套互补证据；
3. 一个由发现导出的 APR 方法，而不是先有模块再找故事；
4. 多数据、多区域、跨姿态和时序验证；
5. 公共 split、脚本、失败 ledger 与计算披露；
6. 至少一次强 negative-control 和一次跨架构复核。

只完成“LAM 上加 residual + 若干平均指标”不够。

