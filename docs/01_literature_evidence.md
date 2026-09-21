# 相关工作证据与创新边界

本表只记录与中心命题直接相关的证据。最终论文应逐项复核版本、许可、训练数据和可复现性。

| 工作 | FLAME / 显式先验扮演的角色 | 与本研究最相关的证据 | 尚未回答的问题 |
|---|---|---|---|
| GGHead (2024) | template UV 排列、局部几何支撑 | Gaussian 可从模板偏移；sphere/plane/FLAME/adjusted-FLAME 消融显示模板规则性与正面采样密度可能比精确 FLAME 形状更重要 | 未拆分 geometry 与 correspondence；无规模交叉曲线 |
| Portrait4D-v2 (2024) | 绕开不准 3DMM reconstruction | 精确合成多视图不足时会出现扁平几何，说明移除先验会把负担转移到数据 | 没有在共同架构下测先验的样本效率与上限 |
| NPGA (2024) | neural parametric prior + per-Gaussian latent dynamics | 说明固定 FLAME expression space 可限制细节，已有 residual/dynamic prior 先例 | 不是四角色因果拆分，也不估计 crossover |
| GAGAvatar (2024) | 2D→3D dual lifting，弱化显式模板依赖 | 提供可训练的另一架构路线 | 不能单独承担 FLAME 因果消融，因为架构同时改变 |
| LVSM (2024/2025) | 最小显式 3D bias 的 Transformer | 规模扩大可产生强 NVS 泛化，支持“数据可能替代部分先验”的大背景 | 不是头像动画控制问题，也未隔离 FLAME 角色 |
| LAM (SIGGRAPH 2025) | canonical vertices、固定 query 对应、LBS/blendshape、tracker labels | 作者明确列出舌头、动态皱纹与 tracking error 限制；公开推理实现有 embedding/flame/e2e_flame query 入口 | 公开仓库缺可直接复现实验的完整训练 runner；renderer 仍会重新注入 FLAME，不能把 embedding 模式当作 no-FLAME |
| PanoLAM (2025) | sparse FLAME points + Transformer + Gaussian densification | 延续“规则稀疏支架降低预测难度”路线 | 代码/数据可用性不足；无作用拆分和 scaling |
| OMEGA-Avatar (2026) | 保拓扑的 FLAME mesh deformation | 已把模板扩张到头发等区域，是“局部释放”近邻 | 未研究四作用、样本效率交叉与 oracle ceiling |
| OMG-Avatar (2026) | FLAME prior + monocular tracker | 明确指出舌头、头发形变与超过约 60° 大姿态局限 | 是方法 limitation，不是系统的 bias phase diagram |
| Any3DAvatar (2026) | FLAME-free structured Gaussian scaffold | 证明无需 FLAME 的生成路线正在成熟 | 数据与控制空间不同，不能据此直接归因 FLAME |
| FFAvatar-N（Nguyen et al., 2026） | FLAME query、local offsets、固定 LBS、端到端 estimator | 约百万身份与大模型强化了 scale 竞争；仍承认眼神、口内、舌头和 FLAME 边界 | 私有大数据；未分离 G/C/A/P，未做公共 crossover |
| HeadsUp (2026) | 固定 neutral template、UV Gaussians、早期 tracked-mesh loss | 已系统研究身份数、输入视图数与容量；固定 neutral template 优于 expression-tracked mesh | 是最接近的 novelty threat；仍未做四作用因果拆分、区域 crossover 与 oracle ceiling |
| MVCHead (2026) | 无参数人脸模型，2D supervision + 3D Gaussian | 展示 2D 数据学习 3D 分布与跨视图一致性的另一端点 | 主任务/数据与可动画头像不完全一致；未测 hard prior crossover |
| FFAvatar-Y（Yao et al., ECCV 2026） | FLAME vertex primitives、UV densification、parametric deformation + residual motion refinement | 已覆盖增量输入视图、sparse-to-dense 和 subject-specific motion residual；与 `A-adaptive` 方法形态高度重叠 | 未把 residual 作为四角色因果 intervention，也未报告 `N_obs × capacity × region` 的 hard/free 符号反转 |
| MATCH (2026) | 固定 template UV + registration-guided attention，学习帧内/跨身份 correspondence | 证明 learned correspondence 与固定 UV template 可以共存；`C-adaptive` 不能仅以“学习对应”作为贡献 | 未单独保持 geometry/animation 不变来测 correspondence release dose，也未研究数据规模 crossover |
| SVG-Head (ICCV 2025) | FLAME-bound surface Gaussians + lips/hair volumetric Gaussians + mesh-aware UV | 已明确采用区域/表示类型混合，说明“FLAME 内外分支”本身不是新颖点 | 目标是重建和编辑；没有作用级 bias phase diagram 或样本效率曲线 |
| PhysHead (CVPR 2026) | parametric head mesh + strand hair + attached Gaussians | 已按物理运动机制给头部和头发分配不同表示，进一步占据区域混合表示路线 | 依赖多视图视频；未回答何时应随数据/容量释放 FLAME 角色 |
| FastGHA (ICLR 2026) | per-pixel free-ish Gaussians + learned expression deformation；另用外部 point-map geometry supervision | 是弱化 FLAME geometry/animation 的强对照，说明 learned dynamics 本身也不是空白 | 同时改变表示、监督和 backbone，不能因果归因哪个 FLAME 角色被替代 |

## 直接推论

1. `FLAME vs. free` 的二元比较不够，因为 geometry、UV/query ordering、animation 和 tracker 会同时变化。
2. “用 residual 释放 FLAME”不够新；APR 必须有作用级 gate、区域级解释、参数匹配和因果证据。
3. HeadsUp 已占据“模板头像 scaling study”的宽泛表述。本研究必须把标题、摘要和实验都锚定在 causal factorization 与 ceiling certificate。
4. FFAvatar-N 的百万身份为规模上限提供竞争参照，但私有数据使公共可复现的中等规模 phase diagram 仍有价值。
5. FFAvatar-Y 已直接实现 parametric deformation 之外的 motion residual；`A-adaptive` 只能作为因果测量工具，不能单独作为方法贡献。
6. MATCH 已实现 learned correspondence，SVG-Head/PhysHead 已实现区域混合表示。剩余可守的新颖性是：在共同 renderer 中逐角色测 `G_mu/G_Sigma/G_density/C/A/P` 的 dose-response、随 `N_obs/N_id/capacity/region` 的交叉曲线，以及由独立证据驱动的联合释放策略。

## 主来源

- GGHead: <https://arxiv.org/abs/2406.09377>
- Portrait4D-v2: <https://arxiv.org/abs/2403.13570>
- NPGA: <https://arxiv.org/abs/2405.19331>
- GAGAvatar: <https://arxiv.org/abs/2410.07971>
- LVSM: <https://arxiv.org/abs/2410.17242>
- LAM: <https://arxiv.org/abs/2502.17796>
- PanoLAM: <https://panolam.github.io/>
- OMEGA-Avatar: <https://arxiv.org/abs/2602.11693>
- OMG-Avatar: <https://arxiv.org/abs/2603.01506>
- Any3DAvatar: <https://arxiv.org/abs/2604.13856>
- HeadsUp: <https://arxiv.org/abs/2605.04035>
- FFAvatar-N（Nguyen et al.）: <https://arxiv.org/abs/2605.15320>
- FFAvatar-Y（Yao et al.）: <https://arxiv.org/abs/2606.30347>
- MVCHead: <https://arxiv.org/abs/2605.25220>
- MATCH: <https://arxiv.org/abs/2603.15811>
- SVG-Head: <https://arxiv.org/abs/2508.09597>
- PhysHead: <https://arxiv.org/abs/2604.06467>
- FastGHA: <https://arxiv.org/abs/2601.13837>
- RenderMe-360: <https://arxiv.org/abs/2305.13353>
