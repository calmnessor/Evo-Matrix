---
date: "2025-01-07"
paper_id: "arXiv:2501.03841"
title: "OmniManip: Towards General Robotic Manipulation via Object-Centric Interaction Primitives as Spatial Constraints"
authors: "Mingjie Pan, Jiyao Zhang, Tianshu Wu, Yinghao Zhao, Wenlong Gao, Hao Dong"
domain: "VLA"
tags:
  - 论文笔记
  - VLA
  - Robotic-Manipulation
  - Object-Centric
  - Interaction-Primitives
  - Zero-Shot
  - VLM
  - Dual-Closed-Loop
  - Self-Correction
quality_score: "8.5/10"
created: "2026-05-15"
updated: "2026-05-15"
status: analyzed
venue: "ICLR 2026"
---

# OmniManip: Towards General Robotic Manipulation via Object-Centric Interaction Primitives as Spatial Constraints

## 核心信息
- **论文ID**：arXiv:2501.03841
- **作者**：Mingjie Pan, Jiyao Zhang, Tianshu Wu, Yinghao Zhao, Wenlong Gao, Hao Dong
- **机构**：CFCS, School of CS, Peking University; PKU-AgiBot Lab; AgiBot
- **发布时间**：2025-01-07
- **会议/期刊**：ICLR 2026
- **链接**：[arXiv](https://arxiv.org/abs/2501.03841) | [PDF](https://arxiv.org/pdf/2501.03841) | [Project Page](https://omnimanip.github.io)
- **引用**：--

## 摘要翻译

### 英文摘要
The development of general robotic systems capable of manipulating in unstructured environments is a significant challenge. While Vision-Language Models (VLM) excel in high-level commonsense reasoning, they lack the fine-grained 3D spatial understanding required for precise manipulation tasks. Fine-tuning VLM on robotic datasets to create Vision-Language-Action Models (VLA) is a potential solution, but it is hindered by high data collection costs and generalization issues. To address these challenges, we propose a novel object-centric representation that bridges the gap between VLM's high-level reasoning and the low-level precision required for manipulation. Our key insight is that an object's canonical space, defined by its functional affordances, provides a structured and semantically meaningful way to describe interaction primitives, such as points and directions. These primitives act as a bridge, translating VLM's commonsense reasoning into actionable 3D spatial constraints. In this context, we introduce a dual closed-loop, open-vocabulary robotic manipulation system: one loop for high-level planning through primitive resampling, interaction rendering and VLM checking, and another for low-level execution via 6D pose tracking. This design ensures robust, real-time control without requiring VLM fine-tuning. Extensive experiments demonstrate strong zero-shot generalization across diverse robotic manipulation tasks, highlighting the potential of this approach for automating large-scale simulation data generation.

### 中文翻译
在非结构化环境中开发通用机器人操控系统是一个重大挑战。视觉语言模型在高层常识推理方面表现出色，但缺乏精确操控任务所需的细粒度3D空间理解能力。在机器人数据集上微调VLM以创建VLA是一个潜在的解决方案，但受到高数据收集成本和泛化问题的阻碍。为了解决这些挑战，我们提出了一种以物体为中心的新型表征，弥合了VLM高层次推理与低精度操控需求之间的差距。我们的核心洞察是：由功能可供性定义的物体规范空间（canonical space）提供了一种结构化且语义上有意义的方式来描述交互原语（如点和方向）。这些原语充当桥梁，将VLM的常识推理转化为可操作的3D空间约束。在此框架下，我们引入了一个双闭环、开放词汇的机器人操控系统：一个闭环负责通过原语重采样、交互渲染和VLM检查进行高层规划，另一个闭环通过6D位姿追踪进行低层执行。该设计在无需VLM微调的情况下实现了鲁棒的实时控制。大量实验证明了在多样化机器人操控任务中的强大零样本泛化能力，并展示了该方法在自动化大规模仿真数据生成方面的潜力。

### 核心要点提炼
- **研究背景**：VLM在机器人操控中缺乏3D空间理解，VLA微调成本高、泛化差，现有原语方法不稳定
- **研究动机**：需要一个高效且可泛化的中间表征来桥接VLM高层推理与低层操控
- **核心方法**：基于物体规范空间的交互原语（点+方向）+ 双闭环系统（规划闭环 + 执行闭环）
- **主要结果**：在12个真实操控任务上显著超越VoxPoser/CoPa/ReKep，闭环规划带来>15%的性能提升
- **研究意义**：为无需VLM微调的零样本开放词汇操控提供了新范式，同时可用于自动化演示数据生成

## 研究背景与动机

### 领域现状
Foundation model在机器人领域的应用迅速发展，特别是VLM在环境理解和高层常识推理方面展现出强大能力。主要技术路线包括：
1. **VLA路线**：在机器人数据集上微调VLM，输出机器人轨迹（如RT-1, RT-2, OpenVLA）
2. **原语提取路线**：利用视觉基础模型提取操作原语，VLM进行高层推理，结合运动规划器执行（如VoxPoser, CoPa, ReKep）

### 现有方法的局限性
1. **VLA方法**：数据收集成本高、耗时长，且微调导致agent特定表征，泛化受限
2. **原语方法**：
   - 原语生成过程是任务无关的（task-agnostic），可能缺乏合适的proposal
   - 将3D原语压缩为VLM所需的2D图像或1D文本存在歧义
   - VLM的幻觉问题使得高层规划不准确
   - 基于关键点的方法在遮挡下不稳定，手动规则后处理引入不稳定性

### 研究动机
如何开发更高效可泛化的表征，在不需要VLM微调的前提下，桥接VLM的高层推理与精确的低层机器人操控？

## 研究问题

核心研究问题：**如何在物体的规范空间（canonical space）中定义交互原语，使其既能为VLM的常识推理提供结构化的语义锚点，又能作为精确空间约束驱动低层动作规划？**

更进一步，**如何构建一个双闭环系统，同时在规划层面（通过自我修正消除幻觉）和执行层面（通过实时位姿追踪）保证鲁棒性？**

## 方法概述

### 核心思想
物体的规范空间由其功能可供性（functional affordances）定义，因此规范空间中的交互点和方向天然地具有语义意义。通过将操控任务形式化为"物体A的交互点/交互方向"与"物体B的交互点/交互方向"之间的空间约束关系，VLM的语义推理可以直接转化为3D空间约束优化问题，由传统算法执行。

### 方法框架

#### 整体架构

![[method_overview.pdf|800]]

> 图2：OmniManip整体架构。给定指令和VFM标注的RGB-D观测，VLM首先筛选任务相关物体并将任务分解为阶段。对每个阶段，VLM以闭环方式提取物体规范空间中的交互原语作为空间约束。执行时，轨迹通过约束优化生成，并利用6D位姿追踪器在闭环中更新。

整个系统流程：
1. **视觉基础模型（VFM）标注**：使用GroundingDINO + SAM标注场景中所有前景物体
2. **VLM任务分解**：筛选任务相关物体，将任务分解为多个阶段 $S = \{S_1, S_2, ..., S_n\}$
3. **3D重建与规范空间建立**：单视图3D生成 + Omni6DPose位姿估计
4. **交互原语提取**：交互点定位 + 交互方向采样
5. **闭环规划（RRC）**：渲染 → VLM检查 → 重采样（自我修正）
6. **闭环执行**：基于约束的轨迹优化 + 实时6D位姿追踪

#### 各模块详细说明

**模块1：任务分解与物体关联**
- **功能**：将自然语言指令分解为有序的操作阶段
- **输入**：RGB图像（VFM标注后）+ 任务指令
- **输出**：阶段序列，每个阶段 $S_i = \{A_i, O_i^{active}, O_i^{passive}\}$
- **处理流程**：
  1. GroundingDINO + SAM标注所有前景物体
  2. VLM（GPT-4o）筛选任务相关物体
  3. VLM将任务分解为子阶段，指定每个阶段的动作、主动物体和被动物体
- **示例**："Pour tea into the cup" → Stage1: Grasp teapot handle; Stage2: Pour tea from teapot into cup

**模块2：交互点定位**
- **功能**：在物体规范空间中定位交互关键点
- **输入**：物体图像 + 3D mesh + 规范空间
- **输出**：交互点 $p \in \mathbb{R}^3$
- **处理流程**：
  1. 使用SCAFFOLD视觉提示机制（叠加Cartesian网格）增强VLM的定位能力
  2. 可见点：直接在图像平面定位
  3. 不可见/不可触摸点：通过多视图推理（从主视角切换至正交视角）
- **关键技术**：
  - 交互点分类：可见可触摸（如把手）vs 不可见或不可触摸（如开口中心）
  - 对于抓取任务：从多个交互点生成heatmap提高鲁棒性

![[method_primitives_points.pdf|600]]

> 图3：交互点生成。按可见性和可触摸性分类，使用SCAFFOLD网格和正交多视图推理。

**模块3：交互方向采样**
- **功能**：在规范空间中确定任务相关的交互方向
- **输入**：物体的规范空间 + 主轴线
- **输出**：排序后的候选交互方向列表
- **处理流程**：
  1. 将物体的主轴（principal axes）作为候选交互方向
  2. VLM为每个候选轴生成语义描述（如"该轴从壶嘴向外延伸"）
  3. LLM评估语义描述与任务的相关性并打分
  4. 按得分排序输出候选方向
- **优势**：相比在SO(3)中均匀采样，基于规范空间主轴的方向采样显著提高了效率（70% vs 30% 在倒茶任务上，且迭代次数更少）

![[method_primitives_directions.pdf|600]]

> 图4：交互方向提取。通过VLM caption + LLM scoring机制排序主轴，选择与任务最相关的方向。

**模块4：闭环规划 RRC（Resampling, Rendering, Checking）**
- **功能**：通过自我修正机制消除VLM幻觉，确保规划正确性
- **输入**：初始原语+约束列表 $K_i$
- **输出**：验证成功的最优约束 $C_i^*$
- **处理流程**：
  1. 初始阶段：按顺序评估约束 $C_i^{(k)}$
  2. **渲染（Render）**：根据当前约束渲染交互场景图
  3. **检查（Check）**：VLM判断渲染结果（成功/失败/需细化）
  4. **重采样（Resampling）**：如果"需细化"，在预测方向 $v_i$ 周围均匀采样6个细化方向
  5. 重新评估，直到找到成功的约束或判定任务失败
- **关键创新**：首次实现规划层面的闭环，而非仅执行层面的闭环

![[closed-loop_planning.pdf|600]]

> 图7：闭环规划示例。"Insert the pen in holder"任务中的自我修正过程。初始规划将笔插入错误的孔，渲染后被VLM识别错误，重采样后修正。

**模块5：闭环执行（Constrained Optimization + Pose Tracking）**
- **功能**：基于空间约束计算并实时调整末端执行器位姿
- **输入**：空间约束 $C$ + 实时物体位姿
- **输出**：末端执行器目标位姿 $P_{ee}$
- **数学公式**：
  $$P_{ee} = \arg\min_{P_{ee}} \sum_{j=1}^{N} L_j(P_{ee})$$
  其中 $L = \{L_C, L_{\text{collision}}, L_{\text{path}}\}$

  - **约束损失**：$L_C = \Delta(C, P_t^{\text{active}}, P_t^{\text{passive}})$，其中 $\Delta$ 衡量当前空间关系与期望约束的偏离，$\Phi$ 将末端执行器位姿映射为主动物体位姿
  - **碰撞损失**：$L_{\text{collision}} = \sum_j \max(0, d_{\min} - d(P_{ee}, O_j))^2$
  - **路径平滑损失**：$L_{\text{path}} = \lambda_1 d_{\text{trans}}(P_t^{ee}, P_{ee}) + \lambda_2 d_{\text{rot}}(P_t^{ee}, P_{ee})$
- **实时追踪**：使用6D位姿追踪器持续更新物体位姿，实现闭环执行

## 实验结果

### 实验设置

#### 硬件配置
- Franka Emika Panda机械臂，UMI手指替换平行夹爪
- 2个Intel RealSense D415深度相机（一个腕部第一人称视角，一个对面第三人称视角）

#### 任务设计
12个开放词汇操控任务：
- **刚性物体操控**（6项）：倒茶、插花入瓶、将笔插入笔筒、回收电池、从碟上取杯、盖上茶壶盖
- **关节物体操控**（6项）：打开抽屉、关闭抽屉、用锤子按按钮、按红色按钮、合上笔记本电脑盖、打开瓶子

每项任务进行10次试验，每次试验后重新配置物体布局。

#### 基线方法
- **VoxPoser**：LLM+VLM生成3D价值图合成轨迹
- **CoPa**：物体部件的空间约束 + VLM
- **ReKep**：关系关键点约束 + 层次优化

### 主要结果

| 任务类型 | VoxPoser | CoPa | ReKep (Auto) | ReKep | OmniManip (Open-loop) | OmniManip (Ours) |
|----------|----------|------|-------------|-------|----------------------|------------------|
| 刚性物体 | 15.0% | 30.0% | 45.0% | 68.3% | 51.7% | **68.3%** |
| 关节物体 | 16.7% | 26.7% | -- | 61.7% | 45.0% | **61.7%** |

### 核心属性分析

#### 原语稳定性
![[primitive_robustness.pdf|600]]

> 图5：交互原语稳定性分析。ReKep通过语义聚类提取关键点但对空间几何不敏感；CoPa对纹理和形状高度敏感；OmniManip在规范空间中采样，确保鲁棒性和任务精度。

#### 视角一致性

![[fig_view_diff.pdf|600]]

> 图6：不同视角下的规划结果对比。ReKep在90°俯视图成功但在0°前视图失败（理想目标点在空气中），OmniManip在规范空间中的表征天然视角不变。

| 方法 | 0° | 25° | 45° | 75° | 90° |
|------|-----|------|------|------|------|
| ReKep | 0/10 | 1/10 | 3/10 | 5/10 | 7/10 |
| OmniManip | 7/10 | 8/10 | 8/10 | 7/10 | 7/10 |

#### 原语采样效率

| 方法 | 回收电池成功率 | 回收电池迭代数 | 倒茶成功率 | 倒茶迭代数 |
|------|---------------|----------------|-----------|-----------|
| Uniform SO(3) | 50% | 1.8 | 30% | 3.4 |
| OmniManip | **80%** | 1.7 | **70%** | 1.8 |

#### 演示数据生成

| 任务 | 行为克隆成功率 |
|------|---------------|
| Pick up cup on dish | 95.24% |
| Recycle battery | 91.30% |
| Insert pen in holder | 86.36% |

## 深度分析

### 研究价值评估

#### 理论贡献
1. **物体中心规范交互原语表征**
   - 创新点：首次将物体规范空间（canonical space）中的交互点和方向作为VLM与低层操控之间的结构化中间表征
   - 学术价值：提供了一种通用的形式化框架，使VLM的语义推理可被精确地转化为3D空间约束
   - 影响范围：VLA、机器人操控、空间推理

2. **双闭环操控系统**
   - 创新点：首次同时实现规划层面（RRC自我修正）和执行层面（6D位姿追踪）的双闭环
   - 学术价值：为缓解VLM幻觉问题提供了一种实用且有效的方法
   - 影响范围：LLM/VLM在机器人中的应用

3. **自动演示数据生成**
   - 创新点：无需任务特定信息即可零样本生成操控演示轨迹
   - 学术价值：为扩展型模仿学习提供低成本数据来源
   - 影响范围：机器人数据收集、Sim-to-Real

#### 实际应用价值
- **零样本开放词汇操控**：无需任何任务特定训练即可执行多样化操控任务，适用于小批量、多品种的工业场景
- **数据生成自动化**：可自动收集演示数据训练模仿学习策略（行为克隆成功率>86%），显著降低人工数据收集成本

### 方法优势详解

1. **效率与效果兼具的原语采样**
   - 基于规范空间主轴而非在SO(3)中均匀采样，将倒茶任务从30%提升至70%成功率，迭代次数从3.4降至1.8

2. **视角不变性**
   - 交互原语定义在物体坐标系中而非图像平面，使得规划结果对相机视角不敏感（ReKep从90°到0°完全失效，OmniManip保持稳定）

3. **无需VLM微调**
   - 所有推理依赖预训练VLM的常识知识（GPT-4o），通过结构化中间表征桥接到3D空间，避免了昂贵的微调

4. **对遮挡的鲁棒性**
   - 基于物体位姿追踪（而非像ReKep基于点追踪），即使交互点被遮挡也能通过物体整体位姿继续追踪

### 局限性分析

1. **形变物体不支持**
   - 描述：无法建模可变形物体（如布料、绳索），因为方法依赖于刚体位姿表征
   - 影响：限制了在涉及柔性物体的任务上的应用
   - 可能方案：结合非刚体追踪或基于粒子的表征

2. **依赖3D AIGC质量**
   - 描述：交互原语提取的有效性依赖于单视图3D重建的mesh质量
   - 影响：对于纹理复杂或结构特殊的物体，mesh质量差可能导致交互点定位不准确
   - 可能方案：使用多视图重建或更高精度的AIGC模型

3. **VLM推理延迟**
   - 描述：多次VLM调用（任务分解、方向描述、RRC检查）带来计算开销
   - 影响：实时性受限，难以直接用于高频控制场景
   - 可能方案：并行处理、模型蒸馏、缓存策略

### 适用性与场景分析

**适用场景**：
- 涉及多个物体的交互任务（如倒茶、插花）
- 需要空间方向约束的精细操作
- 物体种类多样的小批量操控场景

**不适用场景**：
- 涉及柔性/可变形物体的任务
- 需要高频闭环控制的动态任务
- 对实时性要求极高的场景

## 与相关论文对比

### [[ReKep]] - Spatio-temporal Reasoning of Relational Keypoint Constraints

| 对比维度 | ReKep | OmniManip |
|----------|-------|-----------|
| 核心思想 | 关系关键点约束+层次优化 | 物体规范空间交互原语+双闭环 |
| 原语类型 | 图像平面的关键点（2D→3D） | 规范空间中的点+方向（3D原生） |
| 视角不变性 | 差（关键点在图像平面提取） | 好（定义在物体坐标系中） |
| 闭环规划 | 无（仅执行闭环） | 有（RRC自我修正机制） |
| 执行闭环 | 点追踪（47%遮挡失败率） | 位姿追踪（对遮挡更鲁棒） |
| 规划稳定性 | 中 | 高（闭环规划>15%提升） |

### [[CoPa]] - General Robotic Manipulation through Spatial Constraints of Parts

| 对比维度 | CoPa | OmniManip |
|----------|------|-----------|
| 核心思想 | 物体部件的空间约束 | 物体规范空间的交互原语约束 |
| 原语提取 | 基于pixel分割的部件提取 | 规范空间中的功能定义 |
| 对纹理敏感度 | 高（依赖分割质量） | 低（基于功能和几何） |
| 泛化性 | 有限 | 强（零样本跨物体） |

### [[VoxPoser]] - Composable 3D Value Maps

| 对比维度 | VoxPoser | OmniManip |
|----------|----------|-----------|
| 核心思想 | LLM生成3D价值图合成轨迹 | 交互原语空间约束+轨迹优化 |
| 精度 | 粗粒度（价值图） | 细粒度（点+方向约束） |
| 复杂交互 | 有限 | 强（支持方向约束） |

## 技术路线定位

### 所属技术路线
本文属于**VLM驱动的零样本机器人操控（VLM-based Zero-Shot Manipulation）**路线，核心特点：
- 不微调VLM，保留其开放词汇常识推理能力
- 通过结构化中间表征桥接语义理解和空间执行
- 使用传统运动规划器执行精确的低层动作

### 技术路线发展历程
```
VoxPoser (2023) → CoPa (2024) → ReKep (2024) → OmniManip (2025/ICLR2026)
   3D价值图       部件空间约束      关系关键点约束      规范空间交互原语+双闭环
```

### 本文在技术路线中的位置
- **承上**：继承了通过结构化空间约束连接VLM和低层执行的范式
- **启下**：双闭环架构和规范空间原语表征为后续工作提供了更强的可靠性基础
- **关键节点**：从"能用"到"可靠"的关键过渡——通过双闭环解决VLM幻觉和不稳定性

## 未来工作建议

### 作者建议的未来工作
- 无明确建议（论文未详细展开未来工作部分）

### 基于分析的未来方向
1. **结合VLA进行混合策略**
   - 动机：VLA方法正在快速发展，将OmniManip的零样本能力与微调VLA的数据效率结合可能产生互补
   - 可能方法：用OmniManip生成的数据预训练VLA，再用少量高质量人工数据微调

2. **扩展至更多机器人平台与操作类型**
   - 动机：当前仅在单臂操作上验证，扩展到双臂、移动操作等
   - 挑战：多臂协调的空间约束定义、移动基座的额外自由度

3. **提升VLM推理效率**
   - 动机：多次VLM调用影响实时性
   - 可能方法：批量推理、结果缓存、使用更强的开源VLM进行本地部署

## 我的综合评价

### 价值评分

**总体评分**：**8.5/10** — 在零样本操控的可扩展性和可靠性方面做出了重要贡献，双闭环设计和规范空间原语表征具有实际价值。

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 规范空间交互原语+双闭环的组合新颖且有洞察力，但不是范式级别的突破 |
| 技术质量 | 8/10 | 方法设计合理、各组件协同良好，数学形式化清晰 |
| 实验充分性 | 8/10 | 12个真实任务+3个基线+消融+数据生成验证，覆盖较全面 |
| 写作质量 | 8/10 | 结构清晰、图表丰富、逻辑连贯 |
| 实用性 | 9/10 | 零样本即可工作、无需微调、可自动生成演示数据，实用性很强 |

### 重点关注
- **RRC自我修正机制**：通过渲染-检查-重采样实现规划闭环，是解决VLM幻觉的实用方案
- **规范空间原语**：将功能可供性与几何表征统一在物体坐标系中，是通用操控表征的重要探索方向
- **自动数据生成潜力**：OmniManip生成的数据训练BC策略可达>86%成功率，对扩展型机器人学习有重要意义

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 相关论文

### 直接相关
- [[CoPa]] - 同样使用空间约束+VLM做开放词汇操控，但部件级vs物体级
- [[ReKep]] - 同样使用关键点约束，但2D图像平面vs 3D规范空间
- [[VoxPoser]] - LLM+VLM零样本操控的开创性工作

### 背景相关
- [[RT-2]] - VLA模型的代表，通过微调VLM实现机器人控制
- [[OpenVLA]] - 开源VLA模型
- [[Grounded-SAM]] - 视觉基础模型组合，被本文用于物体标注

## 外部资源
- [Project Page](https://omnimanip.github.io) — 项目主页，含视频演示
- [Omni6DPose](https://github.com/... ) — 本文使用的通用6D位姿估计器

> [!tip] 关键启示
> 物体的规范空间（由其功能可供性定义）是连接VLM语义推理与3D操控的最自然桥梁——它既有结构化几何，又有语义意义。

> [!warning] 注意事项
> - 方法依赖GPT-4o的闭源VLM，本地化部署需要替换为开源模型并可能影响效果
> - 3D AIGC质量是关键瓶颈，在复杂或罕见物体上可能失效
> - 不适用于柔性/可变形物体

> [!success] 推荐指数
> 强烈推荐阅读（9/10）—— 如果你是VLA/机器人操控方向的研究者，这篇论文提供了零样本操控的一种优雅且实用的方案。双闭环设计尤其值得借鉴。
