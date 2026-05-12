---
date: "2026-05-12"
paper_id: "arXiv:2409.01652"
title: "ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation"
authors: "Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, Li Fei-Fei"
domain: "VLA_Embodied"
tags:
  - 论文笔记
  - Manipulation
  - Keypoint-Constraints
  - VLM
  - GPT-4o
  - DINOv2
  - Constraint-Optimization
  - Multi-Stage
  - Bimanual
  - In-the-Wild
  - CoRL-2024
  - Foundation-Model
  - Closed-Loop
quality_score: "9.0/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation

## 核心信息
- **论文ID**：arXiv:2409.01652
- **作者**：Wenlong Huang, Chen Wang*, Yunzhu Li*, Ruohan Zhang, Li Fei-Fei（Stanford University, Columbia University）
- **机构**：Stanford Vision and Learning Lab (SVL), Columbia University
- **发布时间**：2024-09-03
- **会议/期刊**：CoRL 2024
- **链接**：[arXiv](https://arxiv.org/abs/2409.01652) | [Project](https://rekep-robot.github.io/)
- **引用**：--

## 摘要翻译

### 英文摘要
Representing robotic manipulation tasks as constraints that associate the robot and the environment is a promising way to encode desired robot behaviors. However, it remains unclear how to formulate the constraints such that they are 1) versatile to diverse tasks, 2) free of manual labeling, and 3) optimizable by off-the-shelf solvers to produce robot actions in real-time. In this work, we introduce Relational Keypoint Constraints (ReKep), a visually-grounded representation for constraints in robotic manipulation. Specifically, ReKep is expressed as Python functions mapping a set of 3D keypoints in the environment to a numerical cost. We demonstrate that by representing a manipulation task as a sequence of Relational Keypoint Constraints, we can employ a hierarchical optimization procedure to solve for robot actions (represented by a sequence of end-effector poses in SE(3)) with a perception-action loop at a real-time frequency. Furthermore, in order to circumvent the need for manual specification of ReKep for each new task, we devise an automated procedure that leverages large vision models and vision-language models to produce ReKep from free-form language instructions and RGB-D observations. We present system implementations on a wheeled single-arm platform and a stationary dual-arm platform that can perform a large variety of manipulation tasks, featuring multi-stage, in-the-wild, bimanual, and reactive behaviors, all without task-specific data or environment models.

### 中文翻译
将机器人操作任务表示为关联机器人与环境的约束，是一种有前景的编码机器人行为的方式。然而，如何构建既通用又多任务、无需人工标注、且能被现成求解器实时优化产生动作的约束表示，仍是一个开放问题。本文提出 Relational Keypoint Constraints（ReKep），一种视觉接地的操作约束表示。ReKep 表达为 Python 函数，将一组 3D 关键点映射到数值代价。我们证明，通过将操作任务表示为 ReKep 序列，可以采用层次化优化过程实时求解机器人动作（SE(3) 末端位姿序列）。此外，为避免每个任务手动指定 ReKep，我们设计了一个自动化流程，利用大视觉模型和视觉语言模型从自由形式语言指令和 RGB-D 观测中生成 ReKep。我们在轮式单臂平台和固定双臂平台上实现了系统，可执行多种操作任务，具备多阶段、野外、双臂和反应式行为，完全无需任务特定数据或环境模型。

### 核心要点提炼
- **研究背景**：操作任务可通过约束表示编码（如相对位姿、学习约束），但现有方法要么不通用、要么需要大量标注数据
- **研究动机**：如何构建一个同时满足"通用性、可扩展获取、实时可解"三条件的约束表示？
- **核心方法**：ReKep = Python 函数（关键点 → 代价），DINOv2 提取关键点 + GPT-4o 生成约束代码 + 层次化优化实时求解
- **主要结果**：自动生成 44.3% 成功率，人工标注 68.6%，支持多阶段/野外/双臂/反应式行为
- **研究意义**：架起 VLM 代码生成能力和机器人操作约束优化之间的桥梁，展示了"VLM 写代码指挥机器人"的新范式

## 研究背景与动机

### 领域现状
操作任务通常涉及复杂的时空约束——例如倒茶任务中，需要"握住把手""保持杯子直立""对齐壶嘴和杯口""以正确角度倾斜"。

现有约束表示方法：
1. **相对位姿约束**：直接但缺乏几何细节、需要物体模型、无法处理可变形物体
2. **数据驱动学习约束**：灵活但随物体/任务组合爆炸式增长，难以扩展

### 研究动机
提出核心问题：**能否设计一种约束表示，同时满足 (1) 广泛适用（多阶段/野外/双臂/反应式），(2) 可扩展获取（通过基础模型自动化），(3) 实时可优化？**

## 研究问题

### 核心研究问题
1. ReKep 作为约束表示的形式定义是什么？
2. 如何将操作任务形式化为 ReKep 约束下的优化问题？
3. 如何高效实时求解？
4. 如何从 RGB-D 和语言指令自动生成 ReKep？

## 方法概述

### 核心思想
**将操作任务表示为"语义关键点之间的数学关系"**。

关键洞见：操作任务的约束本质上是空间中不同实体（机器人、物体部件、其他 agent）之间的空间关系。ReKep 将这些关系表示为 Python 函数——对关键点做算术运算（L2 距离、点积等），输出标量代价（≤0 表示约束满足）。

VLM 不直接操作关键点的数值坐标，而是指定"关系"（如"壶嘴的 z 坐标应大于杯口的 z 坐标"），具体数值在运行时由 3D 跟踪器提供后计算。

### 方法框架

#### 整体架构

![[method.pdf|800]]

> 图1：ReKep 系统概览。DINOv2 提出场景中的语义关键点 → 叠加到 RGB 图像上 → 与指令一起送入 GPT-4o 生成 ReKep 约束 Python 程序 → 层次化优化求解器实时产生末端位姿序列。

#### ReKep 形式定义

单个 ReKep 实例是一个函数：
$$f: \mathbb{R}^{K \times 3} \rightarrow \mathbb{R}$$

将 $K$ 个关键点映射到一个无界代价，$f(\bm{k}) \leq 0$ 表示约束满足。函数实现为无状态 Python 函数，包含 NumPy 操作，可以是非线性/非凸的。

**两类约束**：每个任务分解为 $N$ 个阶段，每阶段 $i$ 包含：
- **子目标约束** $\mathcal{C}_{\text{sub-goal}}^{(i)}$：阶段结束时必须满足
- **路径约束** $\mathcal{C}_{\text{path}}^{(i)}$：阶段内每一时刻必须满足

以倒茶任务为例：
- 阶段 1（抓取）：子目标约束—末端执行器对齐茶壶把手
- 阶段 2（对齐）：子目标约束—壶嘴在杯口上方 10cm；路径约束—保持茶壶直立
- 阶段 3（倒出）：子目标约束—壶嘴在杯口上方 5cm + 壶嘴倾斜

![[teaser.pdf|400]]

> 图2：Teaser — 倒茶任务中 ReKep 的时空约束示意。

#### 层次化优化求解

将操作任务形式化为带 ReKep 约束的优化问题：
$$\argmin_{\mathbf{e}_{1:T}, g_{1:N}} \sum_{i=1}^N \left[ \lambda_{\text{sub-goal}}^{(i)}(\mathbf{e}_{g_i}) + \sum_{t=g_{i-1}}^{g_i} \lambda_{\text{path}}^{(i)}(\mathbf{e}_t) \right]$$
subject to $f(\bm{k}_{g_i}) \leq 0$ 对所有子目标约束，$f(\bm{k}_t) \leq 0$ 对所有路径约束。

**分解求解策略**：
1. **子目标问题**：先求解当前阶段的末端目标位姿 $\mathbf{e}_{g_i}$（Dual Annealing 全局优化 ~1s + SLSQP 局部优化）
2. **路径问题**：从当前位置到子目标的轨迹 $\mathbf{e}_{t:g_i}$（~10Hz 实时局部优化）
3. **回溯**：每控制循环检验路径约束，若违反则回溯到前一阶段

**关键点前向模型**：采用"刚性假设"——末端执行器和"被抓住的关键点组"之间的相对刚性变换仅假设在短时(0.1s)成立。实际关键点位置通过视觉跟踪器 20Hz 更新。

#### 自动 ReKep 生成

![[keypoint-proposal-comparison.pdf|400]]

> 图3：关键点提议流程。DINOv2 提取 patch 特征 → 双线性插值到原图分辨率 → SAM 提取 mask → 对每个 mask 内 DINOv2 特征 K-means 聚类(K=5)→ 聚类中心映射到 3D 世界坐标。

**关键点提议**（DINOv2 + SAM + K-means）：
1. DINOv2 提取 patch-wise 特征 → 双线性插值到原图
2. SAM 提取场景中所有 mask
3. 每个 mask 内 DINOv2 特征余弦相似度 K-means 聚类(K=5)
4. 聚类中心作为关键点候选 → RGB-D 投影到 3D → 8cm 半径去重

**ReKep 代码生成**（GPT-4o Visual Prompting）：
1. 将关键点编号叠加到 RGB 图像上
2. 与语言指令一起送入 GPT-4o
3. GPT-4o 生成：阶段数 + 每阶段的子目标/路径约束 Python 函数
4. 约束函数操作关键点的算术关系（L2 距离、点积等），不直接操作数值

**关键设计优势**：
- VLM 只需指定"关系"而非"数值"（通过算术运算）
- 多个关键点 + 刚性约束 → 可隐式编码完整 SO(3) 旋转
- VLM 不需要处理 3D 旋转的参数化表示（四元数/欧拉角）
- 代码以无状态方式编写，可在优化循环中高效调用

#### 辅助代价函数

子目标求解器包含：场景碰撞避免（Signed Distance Field）、可达性（IK 评分）、位姿正则化、解一致性、双臂自碰撞
路径求解器额外包含：路径长度、轨迹平滑性

抓取阶段利用 AnyGrasp 抓取检测器。

## 实验结果

### 实验平台
- **轮式单臂平台**：移动底座 + 单臂（4 个任务）
- **固定双臂平台**：双 UR5 臂（3 个任务 + 1 个人机协作）

### 任务设计

![[task-viz.pdf|800]]

> 图4：7 个实验任务及优化过程可视化。

| 任务 | 特性 | 挑战 |
|------|------|------|
| Pour Tea | 多阶段/野外/反应式 | 空间对齐+动态约束 |
| Recycle Can | 野外 | 常识知识（可乐罐→回收） |
| Stow Book | 野外 | 狭窄空间运动规划 |
| Tape Box | 野外/反应式 | 物体位姿扰动 |
| Fold Garment | 双臂 | 双臂协调+可变形物体 |
| Pack Shoes | 双臂 | 密集体积约束 |
| Collaborative Folding | 双臂/反应式 | 人机协作+大物体 |

### 主要结果

| 任务 | VoxPoser | ReKep Auto | ReKep Annotated |
|------|----------|------------|-----------------|
| Pour Tea | 0/10 | 3/10 | 8/10 |
| Recycle Can | 3/10 | 6/10 | 8/10 |
| Stow Book | 0/10 | 3/10 | 6/10 |
| Tape Box | 4/10 | 7/10 | 8/10 |
| Fold Garment | 0/10 | 5/10 | 6/10 |
| Pack Shoes | 0/10 | 3/10 | 5/10 |
| Collab. Folding | 0/10 | 4/10 | 7/10 |
| **总计** | **10.0%** | **44.3%** | **68.6%** |

VoxPoser 基线完全无法处理双臂和复杂多阶段任务（除 Tape Box 和 Recycle Can 外全部 0/10）。

### 扰动环境下的鲁棒性

| 任务（扰动） | VoxPoser | ReKep Auto | ReKep Annotated |
|-------------|----------|------------|-----------------|
| Pour Tea | 0/10 | 2/10 | 4/10 |
| Tape Box | 2/10 | 3/10 | 5/10 |
| Collab. Folding | 0/10 | 3/10 | 5/10 |
| **总计** | **6.7%** | **26.7%** | **46.7%** |

快频率关键点跟踪 + 回溯机制使系统能在阶段内和跨阶段重新规划。

### 泛化实验：8 类衣物折叠

![[generalization.pdf|800]]

> 图5：8 种衣物的双臂折叠策略泛化。系统为每种衣物自动生成不同的折叠策略。

| 衣物 | 策略成功率 | 执行成功率 |
|------|----------|----------|
| Sweater | 6/10 | 6/10 |
| Shirt | 4/10 | 5/10 |
| Hoodie | 4/10 | 6/10 |
| Vest | 6/10 | 9/10 |
| Dress | 3/10 | 7/10 |
| Pants | 7/10 | 8/10 |
| Shorts | 7/10 | 9/10 |
| Scarf | 5/10 | 9/10 |
| **总计** | **52.5%** | **73.8%** |

策略成功率（52.5%）测量 ReKep 自动生成的可行性，执行成功率（73.8%）测量给定可行策略后的系统成功率。

### 错误分析

![[error.pdf|400]]

> 图6：系统错误分解。模块按故障贡献率排序。

| 错误来源 | 贡献率 |
|----------|--------|
| 关键点跟踪器 | 最高（间歇遮挡严重） |
| 关键点提议 | 高（遗漏某些关键点） |
| VLM | 高（引用错误关键点编号） |
| 优化模块 | 较低（存在多解空间） |
| 分割/3D 重建/底层控制 | 相对不显著 |

## 深度分析

### 研究价值评估

#### 理论贡献
- **ReKep 约束表示**：将操作任务统一表示为"语义关键点的数学关系"，连接了 VLM 代码生成和约束优化两个领域
  - 创新点：VLM 生成"关系"而非"数值"——通过算术运算指定空间关系，具体数值由 3D 跟踪器提供
  - 学术价值：为 VLM→机器人动作提供了一个新的桥接范式（代码即策略）
- **层次化优化框架**：子目标+路径的分解使得全局规划和局部实时控制可同时实现

#### 实际应用价值
- **零样本操作**：无需任务特定数据或环境模型，仅需 RGB-D + 语言指令
- **跨平台泛化**：同一约束可适用于不同机器人形态
- **可解释性**：约束代码 + 关键点可视化 → 失败可追溯
- **反应式行为**：~10Hz 闭环 + 回溯机制 → 可应对扰动

### 方法优势详解

#### 优势1：代码作为策略表示
- VLM 的代码生成能力 + 优化求解器的可靠性 = 互补优势
- VLM 擅长语义推理（"壶嘴应该在杯口上方"），但不擅长精确数值计算
- 约束优化擅长精确数值计算，但需要形式化的目标
- ReKep 的两阶段设计（VLM 写代码 → 优化器算数值）完美分工

#### 优势2：关键点的语义性
- DINOv2 特征聚类自动发现细粒度语义区域（把手、壶嘴、杯口）
- 无需物体模型或 CAD —— 纯视觉驱动
- 多个关键点组合可表示完整的 3D 姿态约束

#### 优势3：回溯机制
- 不仅在同一阶段内闭环（~10Hz 局部优化），还能跨阶段重规划
- 应对"杯子从手中掉落"等需要回到前一阶段的场景

### 局限性分析

#### 局限1：刚性假设
- 关键点前向模型假设被抓物体与末端执行器刚性连接
- 虽然是"局部"假设（0.1s 窗口 + 20Hz 反馈），但对可变形物体和接触丰富任务有局限

#### 局限2：固定阶段骨架
- 当前假定任务的阶段序列（骨架）是固定的
- 需要不同骨架的重规划需重新调用 VLM（计算开销大）
- 无法在运行中动态插入或删除阶段

#### 局限3：关键点跟踪
- 这是最大的错误来源
- 间歇性遮挡是 3D 视觉跟踪的核心难题
- 依赖 RGB-D 相机 + 校准 → 野外部署受限

#### 局限4：成功率仍有提升空间
- 自动模式 44.3%，扰动下仅 26.7%
- VLM 错误（引用错误关键点编号）和提议遗漏是主要瓶颈

### 适用性与场景分析

#### 适用场景
- **多阶段操作任务**：需要时空依赖的序列操作
- **零样本部署**：新物体/新任务无需训练
- **可变形物体**：衣物折叠等（ReKep 不依赖刚性物体模型）
- **双臂协调**：两条臂之间的约束可自然表达

#### 不适用场景
- **高速动态任务**：~10Hz 不够（投掷、拍打等）
- **精确力控**：约束只关心位姿关系，不涉及力/力矩
- **长时间自主操作**：固定阶段序列 + 无任务级重规划

## 与相关论文对比

### [[pi0_Vision-Language-Action_Flow_Model|π₀]] - VLA Flow Model

#### 方法对比
| 对比维度 | π₀ | ReKep |
|----------|-----|-------|
| 策略表示 | Flow Matching 网络参数 | Python 约束函数 |
| 获取方式 | 大规模预训练+微调 | VLM 零样本生成 |
| 任务数据需求 | 需要演示数据 | 零样本，无需数据 |
| 解释性 | 黑盒 | 白盒（可读代码+关键点） |
| 泛化方式 | 跨 embodiment 预训练 | VLM 常识推理 |

#### 关系分析
- **关系类型**：互补（related）— ReKep 的约束可作为 π₀ 等 VLA 模型的"planning 层"输出
- ReKep 解决"做什么"（约束），VLA 解决"怎么做"（动作生成）
- ReKep 的零样本部署与 VLA 的数据驱动是正交的优势方向

### VoxPoser - 3D Value Map for Manipulation

| 对比维度 | VoxPoser | ReKep |
|----------|----------|-------|
| 表示 | 3D 体素 value map | 关键点约束函数 |
| VLM 输出 | 3D 占用/代价值 | Python 代码 |
| 空间精度 | 体素分辨率受限 | 连续 3D 点，精度更高 |
| 双臂支持 | 不支持（0/10 所有双臂任务） | 支持 |
| 反应性 | 需要重新计算 value map | ~10Hz 闭环优化 |

## 技术路线定位

### 所属技术路线
本文属于 **VLM for Robot Manipulation** 技术路线，核心贡献是"代码即策略"范式——VLM 生成约束代码 + 优化器求解动作。

### 技术路线发展历程
```
Task Planning (LLM) → Code as Policy → VoxPoser (3D VLM) → ReKep (Keypoint Constraints)
    2022-2023          2023-2024         2023                   2024
```

### 本文在技术路线中的位置
- **承上**：继承 VoxPoser 的 VLM-机器人框架，但用关键点约束替代 value map
- **启下**：为"VLM 生成策略代码 + 优化器执行"的范式提供了重要案例
- **关键节点**：从 value map 到 keypoint constraints 的表示升级，解决了双臂和精度问题

## 未来工作建议

### 作者建议
1. 更准确的关键点前向模型（学习或物理驱动）
2. 更鲁棒的关键点跟踪
3. 动态调整阶段序列的重规划能力

### 基于分析的改进方向
1. **与 VLA 集成**：ReKep 输出约束 → VLA 网络生成满足约束的动作序列
2. **学习型关键点提议**：替代启发式 DINOv2+K-means，端到端学习任务相关关键点
3. **约束库复用**：为常见操作（抓取、对齐、倾倒）建立可复用的约束模板
4. **多模态融合**：不仅用 RGB-D，还可集成触觉/力矩反馈作为额外约束
5. **语言→约束→动作端到端**：将 ReKep 作为 VLA 的中间瓶颈表示层

## 我的综合评价

### 价值评分

#### 总体评分
**9.0/10** — 范式创新显著，VLM+优化的互补设计精妙，实验全面覆盖多场景，来自 Fei-Fei 实验室

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9.5/10 | "VLM 写关系+优化器算数值"的互补范式非常巧妙，关键点约束是操作任务的自然抽象 |
| 技术质量 | 9/10 | 系统设计完整（提议→生成→优化→跟踪→回溯），各模块连接清晰 |
| 实验充分性 | 8.5/10 | 7 任务×2 平台+8 衣物泛化+扰动测试+错误分解，但成功率绝对值有提升空间 |
| 写作质量 | 9/10 | 结构清晰，图/表质量高，prompt 开源共享，代码可复现 |
| 实用性 | 8.5/10 | 零样本部署极具吸引力，但自动模式 44.3% 成功率尚需提升才能实际应用 |

### 重点关注
- **"VLM 写关系+优化器算数值"的范式**：这是本文最值得学习的设计哲学——让 LLM 做它擅长的事（语义推理），让优化器做它擅长的事（数值计算）
- **约束作为无状态 Python 函数**：极简表示却具备强大的表达能力（非线性、旋转编码、表面/体积指定）
- **错误分析的文化**：对系统各模块的故障贡献率进行量化分析是难得的工程实践

## 我的笔记

%%
ReKep 最让我印象深刻的是它"不做太多"——VLM 只写关系表达式（如"point_A.y - point_B.y - 0.1"），具体的数值计算交给优化器。这个任务分工非常合理，充分发挥了 VLM 的语义理解和优化器的数值能力。

对比之前分析的 SeqAffordSplat 和 3DAffordSplat，三篇论文从不同角度处理"从指令到操作区域"的问题：
- 3DAffordSplat：识别"可操作区域"（what）
- SeqAffordSplat：识别"按什么顺序操作"（what + order）
- ReKep：识别"操作区域之间的空间约束关系"（how to relate）

如果将它们组合：SeqAffordSplat 输出 affordance mask 序列 → ReKep 将 mask 转化为关键点约束 → VLA 执行满足约束的动作轨迹。这是一个完整的指令→执行 pipeline。

但当前 ReKep 的成功率（自动 44.3%）提示，纯 VLM 零样本方法的可靠性仍不足以替代数据驱动的 VLA。两者结合才是最优路径。
%%

## 相关论文

### 直接相关
- VoxPoser - 前序工作，3D value map + VLM（Stanford/Fei-Fei Lab）
- Code as Policies - 代码作为机器人策略表示的先驱

### 背景相关
- [[3DAffordSplat_Efficient_Affordance_Reasoning|3DAffordSplat]] - 3DGS affordance 识别（关键点可视为 affordance 点的实例）
- [[SeqAffordSplat_Sequential_Affordance_Reasoning|SeqAffordSplat]] - 序列 affordance（ReKep 的 affordance mask 可作为约束输入）
- [[pi0_Vision-Language-Action_Flow_Model|π₀]] - 可执行 ReKep 约束的动作生成
- [[Hi_Robot_Hierarchical_VLA|Hi Robot]] - System 2 推理可与 ReKep 约束互补

### 后续/平行工作
- AnyGrasp - 抓取检测（ReKep 使用的抓取模块）
- DINOv2/SAM - 关键点提议依赖的基础模型

## 外部资源
- [Project Page](https://rekep-robot.github.io/) — 含视频、代码和 prompt

> [!tip] 关键启示
> ReKep 的核心洞察：让 VLM 做"语义关系推理"（写约束），让优化器做"数值计算"（解约束）。这种分工充分利用了各自优势——VLM 不需要精确计算 3D 坐标，优化器不需要理解语义。约束作为 Python 函数的极简表示却具备惊人表达能力。

> [!warning] 注意事项
> - 自动模式成功率 44.3%，扰动下仅 26.7%，距实用还有距离
> - 关键点跟踪是最大瓶颈（间歇遮挡），对 RGB-D 相机和校准依赖强
> - 固定阶段序列骨架，无法运行时动态调整阶段
> - 刚性假设对可变形/接触丰富任务有局限

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！如果你关注 VLM-机器人交互、操作策略表示或基础模型在 embodied AI 中的应用，这是必读论文。ReKep 的"VLM 写代码 + 优化器执行"范式值得所有做"语言到动作"方向的研究者思考。
