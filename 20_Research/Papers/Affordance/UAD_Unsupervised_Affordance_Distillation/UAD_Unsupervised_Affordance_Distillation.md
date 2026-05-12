---
date: "2026-05-12"
paper_id: "arXiv:2506.09284"
title: "UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation"
authors: "Yihe Tang, Wenlong Huang, Yingke Wang, Chengshu Li, Roy Yuan, Ruohan Zhang, Jiajun Wu, Li Fei-Fei"
domain: "Affordance"
tags:
  - 论文笔记
  - Affordance
  - Unsupervised-Learning
  - Foundation-Model
  - DINOv2
  - VLM
  - GPT-4o
  - Imitation-Learning
  - Manipulation
  - Knowledge-Distillation
  - Sim-to-Real
  - Task-Conditioned
quality_score: "8.8/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation

## 核心信息
- **论文ID**：arXiv:2506.09284
- **作者**：Yihe Tang, Wenlong Huang, Yingke Wang, Chengshu Li, Roy Yuan, Ruohan Zhang, Jiajun Wu, Li Fei-Fei（Stanford University）
- **机构**：Stanford Vision and Learning Lab (SVL)
- **发布时间**：2025-06-10
- **会议/期刊**：--
- **链接**：[arXiv](https://arxiv.org/abs/2506.09284) | [Project](https://unsup-affordance.github.io/)
- **引用**：--

## 摘要翻译

### 英文摘要
Understanding fine-grained object affordances is imperative for robots to manipulate objects in unstructured environments given open-ended task instructions. However, existing methods of visual affordance predictions often rely on manually annotated data or conditions only on a predefined set of tasks. We introduce UAD, a method for distilling affordance knowledge from foundation models into a task-conditioned affordance model without any manual annotations. By leveraging the complementary strengths of large vision models and vision-language models, UAD automatically annotates a large-scale dataset with detailed <instruction, visual affordance> pairs. Training only a lightweight task-conditioned decoder atop frozen features, UAD exhibits notable generalization to in-the-wild robotic scenes and to various human activities, despite only being trained on rendered objects in simulation. Using affordance provided by UAD as the observation space, we show an imitation learning policy that demonstrates promising generalization to unseen object instances, object categories, and even variations in task instructions after training on as few as 10 demonstrations.

### 中文翻译
理解物体的细粒度 affordance 是机器人在非结构化环境中根据开放式任务指令进行操作的必要能力。然而，现有 affordance 预测方法通常依赖人工标注数据或仅针对预定义任务集。本文提出 UAD，一种从基础模型中蒸馏 affordance 知识到任务条件化的 affordance 模型的方法，完全不依赖人工标注。利用大视觉模型和视觉语言模型的互补优势，UAD 自动标注大规模 <指令, 视觉 affordance> 数据集。仅在冻结特征之上训练轻量任务条件化解码器，UAD 展现出对野外机器人场景和各种人类活动的显著泛化能力——尽管仅在仿真中渲染的单物体上训练。将 UAD 的 affordance 作为观测空间，我们展示了一个模仿学习策略，仅使用 10 个演示即可泛化到未见物体实例、类别甚至任务指令变化。

### 核心要点提炼
- **研究背景**：机器人操作需要像素级细粒度 affordance 理解，但人工标注扩展性差，现有方法局限于封闭任务集
- **研究动机**：VLMs 在语言空间编码了 affordance 知识，LVMs 提供通用像素级特征——如何将两者互补优势结合起来？
- **核心方法**：DINOv2 多视图融合+聚类→GPT-4o 自动标注 <指令, affordance map> → 冻结 DINOv2+训练轻量 FiLM 解码器 → 作为策略观测空间
- **主要结果**：AGD20K 零样本 SIM=0.407（最优），DROID AUC=0.840，策略<10 demo 即可泛化到新物体/新类别/新指令，真实世界 73% 成功率
- **研究意义**：开创了"无标注 affordance 蒸馏→策略观测空间"的完整范式，展示了基础模型知识向机器人操作迁移的新路径

## 研究背景与动机

### 领域现状
Visual affordance 领域存在两条主要路线：
1. **人工标注路线**（3DAffordanceNet, LASO, 3DAffordSplat）：标注成本高，难以扩展到开放世界
2. **封闭任务路线**：预定义 affordance 类型集合，无法泛化到自由形式指令

另一方面，VLM 在语言空间编码了 affordance 知识（如"把手用于抓握以打开抽屉"），LVMs 提供通用像素级视觉特征——但它们各自有局限：
- VLM：难以将语言知识有效地 grounding 到连续空间域
- LVM：未条件化于特定开放世界任务语义

### 研究动机
核心洞察：**能否利用基础模型（DINOv2 + GPT-4o）自动提取 affordance 标注，蒸馏到一个轻量任务条件化 affordance 模型中，从而在零标注成本下获得泛化性强的 affordance 预测？**

## 研究问题

### 核心研究问题
1. 如何从基础模型中无监督提取细粒度 affordance 标注？
2. 蒸馏得到的 affordance 模型能否泛化到真实世界机器人场景和人类活动？
3. 使用 affordance 作为策略观测空间，能否使模仿学习策略获得泛化能力？

## 方法概述

### 核心思想
**"从基础模型中蒸馏 affordance 知识"**——利用 DINOv2 的细粒度视觉特征和 GPT-4o 的语义推理能力，自动生成大规模 <指令, affordance map> 标注数据，蒸馏到冻结 DINOv2 上的轻量 FiLM 解码器中。在策略层面，使用 affordance map 作为观测空间替代 RGB 图像，使策略天然具备跨物体/类别的泛化性。

### 方法框架

#### 整体架构

![[method.pdf|800]]

> 图1：UAD 方法概览。(a) 多视图 DINOv2 融合+聚类→GPT-4o 任务提议和区域映射→连续 affordance map；(b) 冻结 DINOv2 + 语言条件化 FiLM 层→任务条件化 affordance 模型；(c) affordance map 作为策略观测输入→多视图 transformer 策略。

#### 无监督 Affordance 标注提取

**数据集基础**：BEHAVIOR-1K 的 206 个物体（76 类别）+ 667 任务指令。后期扩展到 Objaverse-XL 10,000+ 物体-指令对。

**步骤 1：细粒度区域提议**

- 对每个 3D 物体，从 14 个视角渲染 RGB 图像 + 聚合点云
- DINOv2 提取逐像素特征 $F_i \in \mathbb{R}^{H \times W \times d}$
- 多视图特征融合（遵循 D3Fields 方法）→ 全局 3D 特征场 $F_{\text{global}} \in \mathbb{R}^{N \times d}$
- PCA 降维（$F_{\text{reduced}} \in \mathbb{R}^{N \times 3}$）减少局部纹理敏感度
- 欧氏距离聚类 → $M$ 个细粒度语义区域（自动确定簇数）

**步骤 2：任务指令提议**

- 选择最佳视角（CLIP 余弦相似度：物体类别名 vs 渲染图）
- 将聚类区域赋予独特颜色叠加到原图上
- 视觉 prompt GPT-4o：叠加图 + 原图 + 物体类别名 → 任务指令集 $\{\mathcal{T}_1, ..., \mathcal{T}_J\}$

**步骤 3：区域-指令映射**

- GPT-4o 将每个任务指令关联到最合适的聚类区域
- 对每个区域计算参考特征 $f_\text{ref}$（区域内点特征均值）
- 计算 $f_\text{ref}$ 与全局特征场 $F_{\text{global}}$ 的余弦相似度 → **连续 affordance map** $A \in [0, 1]^{H \times W}$
- 投影到各相机视角

**关键设计洞察**：VLM 产生离散决策（"这个区域是把手"），而余弦相似度将其转化为**连续**逐像素值——直觉上代表每个像素"afford"指定任务的似然度。这与传统图像级分布归一化（所有像素和为 1）的方法本质不同。

#### 任务条件化 Affordance 模型训练

![[teaser.pdf|800]]

> 图2：Teaser — UAD 从基础模型提取 affordance 标注，蒸馏为任务条件化 affordance 模型，为下游策略提供泛化性。

- **backbone**：冻结 DINOv2 权重
- **解码器**：3 层 FiLM（Feature-wise Linear Modulation），输出通道 [256, 64, 1]
- FiLM 以语言嵌入 $e_{\mathcal{T}}$ 为条件，对 DINOv2 特征做像素位置无关的通道级变换
- 训练只需二元交叉熵损失
- **关键优势**：仅训练轻量解码器 + 冻结 DINOv2 → 在合成单物体上训练，泛化到真实多物体场景

#### 策略学习：Affordance 作为观测空间

- 使用 RVT 的多视图 transformer 策略架构
- 将 affordance map + 深度 + 世界坐标 + 本体感知拼接作为输入
- 策略输出：6-DoF 末端位姿 + 二值夹爪动作
- **核心设计**：affordance 模型在策略训练期间不微调，作为冻结的"任务条件化视觉注意力"前端

## 实验结果

### 实验目标
1. affordance 预测质量（仿真→真实泛化）
2. 策略泛化能力（仿真）
3. 真实世界操作能力

### Affordance 预测评估

#### 仿真渲染物体（Sanity Check）

| 测试集 | AUC |
|--------|-----|
| 训练数据 | ≥0.92 |
| 新物体实例 | ≥0.92 |
| 新类别 | ≥0.92 |
| 新指令 | ≥0.92 |

四类设置均 AUC≥0.92，验证了 UAD 的强泛化能力和与人类判断的一致性。

#### DROID 真实机器人场景

![[droid-viz-short.png|400]]

> 图3：DROID 真实数据集上的 affordance 预测对比。UAD 提供细粒度、连续、鲁棒的特征，聚焦于交互区域。

| 方法 | AUC |
|------|-----|
| CLIP | 0.500 |
| OpenSeeD (开放词汇分割) | 0.836 |
| **UAD** | **0.840** |

UAD 的优势在于：
- **连续**而非二值输出
- 对小物体/小部件的**一致预测**（分割模型的典型失败案例）
- 尽管仅在**合成单物体**上训练，在野外**多物体杂乱场景**中仍表现优异

#### AGD20K 人类活动 Affordance（零样本）

![[agd.pdf|400]]

> 图4：AGD20K 人类活动 affordance 零样本泛化。UAD 可识别"坐自行车""打字""吃香蕉"等完全未见的活动/物体。

| 方法 | KLD↓ | SIM↑ | NSS↑ | NSS-0.5↑ |
|------|------|------|------|-----------|
| LOCATE | **1.405** | 0.372 | **1.157** | 1.723 |
| AffordanceLLM | 1.463 | 0.377 | 1.070 | -- |
| **UAD** | 1.878 | **0.407** | 1.092 | **2.050** |

- SIM 最高（分布相似度最优）
- NSS-0.5 大幅领先（严格阈值下最显著区域匹配最好）
- KLD 较高是因为 UAD 预测更细粒度，而 AGD20K 的 GT 分布更扩散

### 策略泛化评估（仿真）

![[exp_fig.png|400]]

> 图5：三个仿真任务（Pouring/Opening/Insertion）在四种泛化设置下的平均成功率。

**四种泛化设置**：
1. **新位姿**：物体随机放置
2. **新实例/模型**：同类别不同形状物体
3. **新类别**：功能结构相似的不同类别（如啤酒瓶→可乐罐）
4. **新指令**：操作对象变化（如倒水→浇花）

对比基线：RGB、DINOv2、CLIP、Voltron 作为观测空间。

- UAD 在所有泛化设置下均优于基线
- 在需要细粒度视觉感知的 Opening 任务上优势尤为明显（检测抽屉把手的精确抓取点）
- 可通过语言指令灵活切换操作对象

### 真实世界操作

![[sim-real.pdf|800]]

> 图6：仿真和真实世界操作任务及成功率。

| 平台 | 任务数 | 演示数/任务 | 平均成功率 |
|------|--------|------------|-----------|
| Franka Emika Panda | 3（Opening/Pouring/Insertion） | 10 | **73%** |

仅 10 个演示即可实现真实世界操作泛化。

## 深度分析

### 研究价值评估

#### 理论贡献
- **无监督 Affordance 蒸馏范式**：首次证明可以从基础模型中零标注成本提取高质量 affordance 标注
  - 创新点：GPT-4o 离散决策→余弦相似度连续化的巧妙转换
  - 学术价值：为"基础模型知识→机器人感知"提供了一个可扩展的方案
- **Affordance 作为观测空间**：证明 affordance 是比通用视觉特征（CLIP/DINOv2）更有效的策略观测表示
  - 洞察：任务条件化的细粒度注意力天然具备跨物体泛化性

#### 实际应用价值
- **零标注成本**：完全消除 affordance 人工标注瓶颈
- **极低数据需求**：仅需 10 个演示即可泛化到新物体/类别/指令
- **即插即用**：冻结的 affordance 模型可插入任意视觉策略架构

### 方法优势详解

#### 优势1：连续 Affordance 表示
- 传统方法将 affordance 视为二值分割（属于/不属于某功能区域）
- UAD 的连续表示 $[0,1]$ 捕获了"接近功能中心的程度"——例如把手中间比把手尖端更适合抓握
- 连续表示提供更多信息给下游策略（用于加权注意力或动作精度调节）

#### 优势2：冻结 DINOv2 + 轻量 FiLM
- DINOv2 冻结保留预训练特征的通用性 → 仅在合成数据上训练也能泛化到真实场景
- FiLM 条件化于语言嵌入，实现像素位置无关的通道变换 → 学会的是"哪些特征维度与特定任务相关"
- 参数量极小，训练/推理高效

#### 优势3：完整的评估链
- Affordance 预测：仿真 → DROID 真实机器人 → AGD20K 人类活动（三级泛化）
- 策略：仿真四类泛化 → 真实世界操作
- 展现了从基础模型知识到机器人动作的完整迁移路径

### 局限性分析

#### 局限1：仅静态单帧
- affordance 预测基于单帧图像，无法处理时序多步理解
- 操作任务通常涉及多步视觉理解和行为
- 与 SeqAffordSplat/ReKep 等支持序列推理的方法形成互补

#### 局限2：训练数据为合成单物体
- 仅在渲染的单一物体上训练，虽然泛化到多物体场景，但存在分布差异
- 扩展到真实多物体图像标注可能进一步增强 ground 能力

#### 局限3：运动级泛化
- affordance 模型本身不提供运动级泛化——策略仍需在特定任务上训练
- affordance 降低了策略学习的难度，但未完全消除训练需求

### 适用性与场景分析

#### 适用场景
- **细粒度操作任务**：需要精确识别抓取点、对齐区域等
- **开放世界泛化需求**：新物体、新类别、新指令的运行
- **数据稀缺场景**：仅少量演示即可部署

#### 不适用场景
- **时序多步推理**：需要与 SeqAffordSplat/ReKep 结合
- **高速动态任务**：affordance 为静态帧预测，无动态预判

## 与相关论文对比

### [[3DAffordSplat_Efficient_Affordance_Reasoning|3DAffordSplat]] - 3DGS Affordance

#### 方法对比
| 对比维度 | 3DAffordSplat | UAD |
|----------|---------------|-----|
| 表示 | 3D Gaussian（3DGS） | 2D 像素 + 3D 点云特征 |
| 标注方式 | 人工标注（6,631 个） | 完全无监督 |
| 模型输出 | 3D affordance mask | 2D 连续 affordance map |
| 策略集成 | 未涉及 | 作为策略观测空间的端到端验证 |

#### 关系分析
- **关系类型**：互补（related）— UAD 的 2D affordance 可投影到 3DGS 场景中，3DAffordSplat 的 3DGS 表示可增强 UAD 的空间精度
- UAD 的优势在于零标注，3DAffordSplat 的优势在于 3D 一致性

### [[SeqAffordSplat_Sequential_Affordance_Reasoning|SeqAffordSplat]] - 序列 Affordance

| 对比维度 | SeqAffordSplat | UAD |
|----------|----------------|-----|
| 任务范围 | 场景级序列 | 静态单帧 |
| LLM 使用 | Qwen3 自回归生成 | GPT-4o 标注 |
| 策略集成 | 未涉及 | 已集成并验证泛化 |

### [[rekep_2409.01652|ReKep]] - 关键点约束

| 对比维度 | ReKep | UAD |
|----------|-------|-----|
| 策略表示 | 关键点约束→优化求解 | Affordance map→模仿学习 |
| VLM 角色 | 写约束代码（运行时） | 标注数据（离线） |
| 泛化路径 | VLM 常识推理 | 冻结 DINOv2 通用特征 |
| 共同点 | 同一实验室（Stanford SVL），共享多位作者 | |

### 关系网
三篇 Stanford SVL 论文形成了一条完整的"感知→规划→执行"管道：
1. **UAD**：识别 affordance（感知 what+where）
2. **ReKep**：制定约束（规划 how to relate）
3. **策略**：执行动作（执行具体操作）

## 技术路线定位

### 所属技术路线
本文属于 **Visual Affordance for Manipulation** 技术路线，核心贡献是"从基础模型中无监督蒸馏 affordance 知识"。

### 技术路线发展历程
```
人工标注 Affordance → 半监督/弱监督 → 基础模型零样本（VLM直接推理） → 无监督蒸馏（UAD）
  2021-2024              2024               2024                         2025
```

### 本文在技术路线中的位置
- **承上**：继承 DINOv2 的通用视觉特征、GPT-4o 的语义推理、D3Fields 的多视图融合
- **启下**：为"affordance 作为策略观测空间"这一范式提供了完整案例
- **关键节点**：从"需要标注"到"零标注"的转折点，但仍有合成→真实的模拟偏差

## 未来工作建议

### 作者建议
1. 扩展到多步时序 affordance 理解
2. 在真实多物体图像上扩展训练数据
3. 运动级泛化的策略学习

### 基于分析的改进方向
1. **与 3DGS 融合**：将 UAD 的 2D affordance 投影到 3DGS 场景中，实现空间一致的多视图 affordance
2. **时序扩展**：结合 SeqAffordSplat 的序列推理，从单帧 affordance 扩展到操作序列 afffordance
3. **策略-感知联合优化**：当前 affordance 模型冻结不参与策略训练→端到端微调可能进一步提升
4. **触觉/力反馈 affordance**：视觉 affordance 之外，蒸馏"触觉 affordance"（如柔软度、重量）可能进一步提升操作成功率

## 我的综合评价

### 价值评分

#### 总体评分
**8.8/10** — "零标注 affordance 蒸馏→策略泛化"的完整范式，工程实用性强，泛化验证充分

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | 首次实现完全无监督的 affordance 蒸馏+策略集成，离散标注→连续 map 的转换巧妙 |
| 技术质量 | 8.5/10 | 方法设计简约有效（冻结 DINOv2+FiLM），但训练数据规模有限（仅 206 物体） |
| 实验充分性 | 9/10 | 三级泛化评估（仿真→DROID→AGD20K）+四级策略泛化+真实操作，覆盖全面 |
| 写作质量 | 8.5/10 | 结构清晰，图/表设计好，技术细节完整 |
| 实用性 | 8.5/10 | 零标注+10 demo+即插即用，实用性强，但尚未开源到产品级 |

### 重点关注
- **余弦相似度连续化**：将 VLM 离散决策转换为连续 affordance map 的技巧——简单但极为有效
- **"仅合成单物体训练→真实多物体泛化"**：冻结 DINOv2 的通用性在这里发挥了关键作用
- **与 ReKep 的互补性**：同一实验室的两篇论文分别解决了 affordance 的"感知"和"约束求解"两个维度

## 我的笔记

%%
UAD 最让我印象深刻的是它的"减法"设计哲学：
1. 不做人工标注（省掉最大成本）
2. 不训练 backbone（冻结 DINOv2，省掉最大计算）
3. 不微调 affordance 模型（策略训练时冻结）
每一步都在"做减法"，但最终的泛化能力反而更强。这验证了一个重要假设：基础模型的知识已经足够好，我们只需要"蒸馏"而非"重新学习"。

把 UAD、SeqAffordSplat、ReKep 三篇论文放在一起看很有意思：
- UAD 回答"在哪里操作"（affordance map）
- SeqAffordSplat 回答"按什么顺序操作"（sequential）
- ReKep 回答"操作之间的关系是什么"（constraints）

如果把三者组合成一个完整的 instruction-to-action pipeline，会非常强大。但当前三篇论文来自不同实验室（Stanford SVL vs SYSU vs Xidian），尚未看到整合的尝试。

另外，UAD 讨论了一个很关键但常被忽视的点：affordance 应该是连续的而非二值的。把手中间比把手尖端更"可抓握"——这个直觉在连续表示中得以保留，在二值 mask 中则丢失了。下游策略可能利用这些细粒度差异来优化抓取位姿。
%%

## 相关论文

### 直接相关
- D3Fields - 多视图 DINOv2 特征融合（UAD 的区域提议基础）
- [[rekep_2409.01652|ReKep]] - 同一实验室，关键点约束操作（互补范式）
- LOCATE/AffordanceLLM - AGD20K 基准竞争对手

### 背景相关
- [[3DAffordSplat_Efficient_Affordance_Reasoning|3DAffordSplat]] - 3DGS affordance（3D vs 2D 表示互补）
- [[SeqAffordSplat_Sequential_Affordance_Reasoning|SeqAffordSplat]] - 序列 affordance（时序扩展方向）
- [[pi0_Vision-Language-Action_Flow_Model|π₀]] - VLA 可集成 UAD 的 affordance 作为感知前端
- DROID - 真实机器人数据集（UAD 的评估基准）

### 后续/平行工作
- RVT - 多视图 transformer 策略（UAD 的策略架构基础）
- DINOv2 - 视觉 backbone
- CLIP - 用于对比的语言-图像模型

## 外部资源
- [Project Page](https://unsup-affordance.github.io/) — 含代码和数据

> [!tip] 关键启示
> 基础模型的知识已经足够好——我们不需要"重新学习"affordance，只需要"蒸馏"。冻结 DINOv2+训练轻量 FiLM 是性价比极高的设计。离散 VLM 决策→连续余弦相似度 map 的转换是连接"语义推理"和"像素级感知"的优雅桥梁。

> [!warning] 注意事项
> - 仅在合成单物体上训练，复杂多物体交互场景的 affordance 重叠问题未处理
> - 静态单帧 affordance，无法直接处理操作序列
> - 策略仍需少量演示训练（10 demo），不能完全零样本操作
> - 依赖 RGB-D 相机，无深度传感器场景 afffordance 投影受限

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！无论你关注 affordance learning、基础模型应用还是 robot manipulation，UAD 都提供了一种简约而高效的范式。特别是"冻结 backbone + 轻量条件化解码器"的设计值得所有做 robot perception 的研究者参考。
