# FLAME 四角色因果拆分

## 变量定义

对每个方法，用四元组 `(G,C,A,P)` 描述 FLAME 介入方式。每一维至少有三种状态：

| 作用 | Hard | Soft / residual | Free / independent |
|---|---|---|---|
| `G` geometry | query 坐标限制在 FLAME surface/小邻域；同时声明 covariance 与 densification 上限 | `mu = mu_F + rho_mu * delta_mu`，并独立控制 extent | 坐标由共享 scaffold 或 learned slots 预测，不读取 FLAME 坐标 |
| `C` correspondence | 固定 FLAME vertex/UV ID 与邻域 | 语义 anchor 上的局部 soft transport | 无 FLAME 语义 ID；只保留方法自身 permutation-consistent slots |
| `A` animation | FLAME LBS + blendshapes 是唯一 deformation | FLAME transform + residual SE(3)/flow | 由相同观测控制信号驱动的 learned deformation，不读取 FLAME skinning/blendshape |
| `P` pseudo-label | tracker pose/expression/camera 固定为监督 | 以 tracker 为均值，学习置信度加权 correction | GT calibration/self-calibration/latent control，不把 tracker 输出作为标签 |

`P` 是监督偏置，不是表示偏置。主分析必须把 `P` 与 `G/C/A` 分开报告。

## 最小可识别干预

### Geometry `G`

在 Gaussian 表示中先拆成三个子通道：

- `G_mu`：Gaussian center/query mean 是否锚定 FLAME；
- `G_Sigma`：协方差、最大轴标准差和方向是否可把支撑延伸到模板外；
- `G_density`：Gaussian 数、split/clone/densification 是否在模板外产生新支撑。

已完成的 `G_mu × G_Sigma × G_density` 因子实验表明，大 extent 会在 hard means 下显著提高 RGB，增加 anchored splats 还会放大这一补偿路径，但 center geometry 只发生极小变化。因此任何 mean-release ladder 都必须报告 scale 分布、axis cap、Gaussian 数和 geometry metric；将“mean hard”直接称为“geometry hard”是不准确的。

保持 slot 数、slot ID、decoder、animation、label 完全相同，只替换 canonical coordinate：

1. `G-hard`：FLAME point，offset 上限很小；
2. `G-bounded`：相同 FLAME point，逐级增加 offset 半径；
3. `G-residual`：不设固定半径，以正则控制偏移；
4. `G-free`：learned coordinate，但 slot ID 和 decoder 顺序不变。

这条 ladder 估计几何约束强度的剂量效应，而不是只做两个端点。

每个 rung 再与至少两种 `G_Sigma` cap 和 parameter-matched densification control 交叉。Gaussian scale 在本仓库定义为 principal-axis standard deviation，不能把 30 mm cap 误读成仅 30 mm 总覆盖范围。

### FLAME-semantic correspondence `C`

保持坐标分布、Gaussian 数和 decoder 相同：

1. `C-hard`：固定 vertex/UV ID 和 mesh neighborhood；
2. `C-local`：允许在 `k` 个相邻 anchor 间做 soft assignment；
3. `C-global`：允许在全局 semantic anchors 上 transport；
4. `C-free`：移除 FLAME semantic ID，使用无 FLAME 语义的 learned slots。

learned slots 仍可能形成隐式对应，因此准确表述是“移除 **FLAME-semantic correspondence**”，不能声称“没有 correspondence”。

当前已完成一个 identification intervention：逐帧 Gaussian centers、animation 和 tracking 完全不变，只重排 attribute identity。local 条件在 10 mm canonical bins 内重排，global 条件全局重排。它证明 C 对 shared-attribute renderer 有独立效应，并给出“local 明显优于 global”的中间态证据；但它是 corruption test，不是 `C-local soft transport` 方法结果。

### Animation `A`

所有方法得到相同的输入证据，例如图像 encoder feature、音频或 driving-frame latent；否则 FLAME-free 方法无法表达舌头等未进入 FLAME 系数的运动，比较不公平。

1. `A-hard`：只用 FLAME LBS/blendshape；
2. `A-residual`：`T_i = T_i^FLAME compose Exp(rho_A * xi_i(z_dyn))`；
3. `A-free`：相同 `z_dyn` 驱动 learned deformation field，不读 FLAME skinning/blendshape。

当前动态 oracle 使用官方 tracking code 的 PCA-conditioned shared deformation basis 作为共同 `z_dyn`：hard 读 FLAME LBS/blendshape；free 从 neutral identity mesh 学 deformation；adaptive 在 hard 上加逐顶点 bounded gate。五身份 observation-scale 与 capacity curves 已完成。由于公开训练数据只有单 camera，当前只主张 temporal generalization。

### Pseudo-label `P`

1. `P-hard`：冻结 tracker camera/pose/expression；
2. `P-corrected`：tracker 为均值，按置信度学习 correction；
3. `P-selfcal`：多视图 calibration 固定，pose/expression 与 avatar 联合优化；
4. `P-oracle`：在合成数据或有可靠标定的数据上使用真值。

`P` 的干预先在固定 `G/C/A` 下执行。如果 `P-selfcal` 消除性能平台，结论应写成 supervision bottleneck，而不是 representation bottleneck。

当前完成的是 controlled timing intervention：对 official tracking 整体施加 1/2/4-frame lag，同时冻结 clean RGB/alpha 和 clean evaluation ROI。它识别 P timing error 的独立剂量效应，但不是自然 tracker error audit，也不是 `P-corrected/P-selfcal` 方法比较。发现 adaptive A 对 lag 的相对 clean drop 比 hard 更大，因此未来 APR 必须显式包含 `rho_P` 或 confidence correction。

## 识别风险与控制

| 风险 | 必须控制的方法 |
|---|---|
| FLAME 坐标与 vertex ID 天然绑定 | 用相同 slot ID 替换坐标；另做坐标分布匹配的随机/球面/平面 scaffold |
| free 方法参数更多 | 所有分支计入总参数；加 parameter-matched hard/free 控制 |
| free 方法训练更难 | 报告相同 step 和收敛预算两套结果；oracle-fit 用多初始化和加长预算 |
| FLAME-free 缺少控制信号 | 全部方法共享 `z_dyn`；只改变 deformation mechanism |
| tracker 同时影响输入与评价 | 主评价使用 FLAME-independent mask、标定、深度/点云或人工区域；FLAME metric 仅为次要结果 |
| 神经 refiner 掩盖 3D 错误 | 主表以 renderer 直接输出为准；refiner 结果单列并配 geometry/consistency |
| region mask 随方法变化 | mask 在 GT 图像/独立分割器上冻结，映射规则预注册 |

## 完整 factorial 与可行子集

全 `3^4` factorial 太贵，也包含难解释的组合。采用两级设计：

- 主曲线：`hard=(0,0,0,0)`、`free=(2,2,2,2)`、`APR=learned gates`；
- 角色识别：在低/中/高数据各选一处 anchor，做 `G/C/A/P` 单角色 release；
- 交互：只预注册 `G×C`、`G×A`、`A×P` 三组，因为它们有明确机制；
- oracle-fit：固定 `P=oracle/selfcal`，完整扫描 `G/C/A` 的必要端点。

## APR 参数化

APR 不是简单添加 residual，而是学习局部先验精度：

```text
mu_i = mu_i^FLAME + rho_i^G * delta_mu_i
Sigma_i = Release(Sigma_i^base; rho_i^Sigma, Sigma_max)
q_i  = Transport(q_i^FLAME; rho_i^C, P_i)
T_i  = T_i^FLAME compose Exp(rho_i^A * xi_i(z_dyn))
y    = y_tracker + rho^P * delta_y
```

其中 `rho in [0,1]` 由区域 feature、重建残差、跨视图冲突、tracker confidence 和训练规模 token 条件化。初始化 `rho=0`，用 release budget / prior precision penalty 防止一开始退化成 free 模型。

必须比较：

- hard；
- free；
- 固定 residual 强度；
- 全局 scheduled release；
- 全局 learned gate；
- region-aware APR；
- 参数匹配的 free residual control。

若 APR 的 `rho` 与独立 oracle gap、区域误差或 tracker confidence 不相关，它只是一个额外网络，不能支撑“adaptive prior release”的机制结论。

## 当前角色证据状态

| role | 已完成 | 尚缺 |
|---|---|---|
| `G` | `G_mu × G_Sigma × G_density`、held-out-view release、五案例复现 | 动态 multi-view、跨身份 learned geometry |
| `C` | centers-fixed local/global attribute-identity dose | learned local transport、free implicit slots |
| `A` | hard/adaptive/free observation scale、capacity、四 motion families | dynamic novel-view、tongue、跨身份 backbone |
| `P` | clean-ROI controlled tracking lag | natural error distribution、corrected/selfcal/oracle ladder |

这张表的目的不是把 corruption test 等同于完整方法，而是明确当前每个因果接口已经识别到什么、还没有解决什么。
