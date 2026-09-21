# Paper blueprint

## 核心叙事

1. 参数人头先验在小数据区间降低估计误差，但它的不同组成会产生不同的区域上限。
2. 现有方法把 FLAME 的四种作用绑在一起，因而无法判断性能来自几何、对应、动画还是 tracker。
3. oracle-fit 给出 ceiling certificate，scaling curve 给出实际 crossover，二者共同构成 prior phase diagram。
4. APR 根据独立冲突证据释放对应作用，获得更好的质量–数据–算力–控制 Pareto 前沿。

## 预期主图

### Fig. 1 — Prior phase diagram

横轴身份/观测规模，纵轴容量，分面为 stable skin、oral/tongue、hair/dynamic detail；颜色表示 `hard-free` 配对差，等高线是 crossover 与 CI/删失。

当前可落图部分分两列：左为五身份 `A` 的 `8/24/full × K={4,8,16,32,64}` oral heatmap；右为 20 held-out identities 上的 `G_mu` nested `N_id={8,32,128,384}` 曲线。右列至少分 head-supported、head-unsupported、head-foreground，并用 full-target/head-target 两个 panel 显示 crossover 会随优化目标移动。硬 unsupported 平台、release scale gain 与 CI 已齐。

### Fig. 2 — Causal factorization

一条共同 pipeline，只在四个明确接口切换 `G/C/A/P`。图中标出 control signal 和 evaluator，避免把 tracker 同时放在输入与真值两侧。

### Fig. 3 — Geometry-channel identification

左侧画 `mean release radius → converged error`；中间画 `mean anchoring × covariance extent × density` 的三通道交互；右侧并列 RGB、center geometry 与 held-out view。当前五案例结果已显示：hard density 只有在 wide extent 下明显增益，而 4× 后 unsupported center error 仍约 35.84 mm。先证明更多/stretched splats 能改善图像却不修复支撑面，再给出受控 ceiling certificate。

### Fig. 4 — Scaling crossover

hard/free/APR 在 `N_id × capacity` 下的原始 seed 点、均值、CI；身份数与每身份观测量分别画。

Fig. 4a 用五身份 observation-scale spaghetti：free−hard oral 由 8 帧全负变为 full 全正，标题明确 observation count。Fig. 4b 用真正的跨身份 `N_id` scaling：36-run primary、20 identities、seed/identity paired CI。Fig. 4c 放 capacity 64/128/512、20k/60k、sigma 3/5 mm 的 released-hard gap forest plot；强调 capacity/convergence 稳健而 covariance 移动边界。

### Fig. 5 — Role attribution

`G/C/A/P` 单角色释放和 `G×C/G×A/A×P` 交互的效应量 forest plot；把 representation 与 supervision bottleneck 分开。

当前素材：

- `G`：mean×extent×density 三通道 interaction、绝对 center-error floor 与 held-out-view regional Pareto；
- `C`：centers-fixed local/global identity-shuffle dose；
- `A`：observation/capacity crossover 与 motion-region interaction；
- `P`：tracking-lag dose，以及 adaptive-drop minus hard-drop 的负 interaction。

`C/P` 当前是 controlled corruption，不应与 learned release method 放在同一颜色/标题下冒充主方法消融。

### Fig. 6 — APR mechanism and Pareto

局部 `rho_G/rho_C/rho_A/rho_P` trust maps、与独立 oracle/tracker conflict 的相关性，以及参数匹配 Pareto 图。

当前已有两类可画 Pareto：A-only gate 的 oral 非支配点，以及跨身份 `G_mu` adaptive radius 30/75/150 的 quality–displacement curve。`N_id=384` 的 r=30 高于 hard→released chord，但 LPIPS 不支配 hard，图中必须并列 PSNR 与 LPIPS，不能只展示成功指标。完整 `rho_G/rho_C/rho_A/rho_P` gate correlation 仍是 method extension，不把它冒充已完成结果。

## 主表

- Table 1：相关方法是否使用 `G/C/A/P`、训练规模、是否公共；
- Table 2：oracle-fit region metrics；
- Table 3：公共 benchmark 主结果与 yaw/expression stress；
- Table 4：sample efficiency、GPU-hours、参数、Gaussian 数、latency；
- Table 5：APR 与 fixed residual/global schedule/parameter-matched controls。
- Table 6：target-region 与 1×/2× density mechanism controls，分别报告 method absolute delta 与 released-hard gap delta。

## 摘要必须包含的限定

- crossover 是 region- and role-dependent，不声称一个全局阈值；
- “ceiling”由 oracle + scaling 共同支持；
- tracker 污染单列为 supervision bias；
- 大规模私有数据只作参照，主要发现可在公共 protocol 复现。

## 最容易被审稿人击穿的三句话

以下说法在没有对应证据前禁止出现：

- “我们首次放松 FLAME 先验”；
- “更多数据自然消除对 3DMM 的需求”；
- “APR 学会在需要时释放先验”。

它们分别需要完整相关工作边界、观测范围内的统计曲线、以及 gate 与独立冲突证据的机制验证。
