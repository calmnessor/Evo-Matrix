---
date: "2023-09-18"
paper_id: "arXiv:2309.09818"
title: "Grasp-Anything: Large-scale Grasp Dataset from Foundation Models"
authors: "An Dinh Vuong, Minh Nhat Vu, Hieu Le, Baoru Huang, Binh Huynh, Thieu Vo, Andreas Kugi, Anh Nguyen"
domain: "Grasp"
tags:
  - 论文笔记
  - Grasp
  - Dataset
  - Foundation-Model
  - ChatGPT
  - Stable-Diffusion
  - Zero-Shot
  - Data-Centric
quality_score: "7.5/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# Grasp-Anything: Large-scale Grasp Dataset from Foundation Models

## 核心信息
- **论文ID**：arXiv:2309.09818
- **作者**：An Dinh Vuong, Minh Nhat Vu, Hieu Le, Baoru Huang, Binh Huynh, Thieu Vo, Andreas Kugi, Anh Nguyen
- **机构**：University of Liverpool / TU Wien (基于作者所属推断)
- **发布时间**：2023-09-18
- **会议/期刊**：ICRA 2024
- **链接**：[arXiv](https://arxiv.org/abs/2309.09818) | [Project Page](https://airvlab.github.io/grasp-anything/)
- **引用**：[DOI: 10.1109/ICRA57147.2024.10611277](https://doi.org/10.1109/ICRA57147.2024.10611277)

## 摘要翻译

### 英文摘要
Foundation models such as ChatGPT have made significant strides in robotic tasks due to their universal representation of real-world domains. In this paper, we leverage foundation models to tackle grasp detection, a persistent challenge in robotics with broad industrial applications. Despite numerous grasp datasets, their object diversity remains limited compared to real-world figures. Fortunately, foundation models possess an extensive repository of real-world knowledge, including objects we encounter in our daily lives. As a consequence, a promising solution to the limited representation in previous grasp datasets is to harness the universal knowledge embedded in these foundation models. We present Grasp-Anything, a new large-scale grasp dataset synthesized from foundation models to implement this solution. Grasp-Anything excels in diversity and magnitude, boasting 1M samples with text descriptions and more than 3M objects, surpassing prior datasets. Empirically, we show that Grasp-Anything successfully facilitates zero-shot grasp detection on vision-based tasks and real-world robotic experiments.

### 中文翻译
ChatGPT 等基础模型凭借其对真实世界领域的通用表征，已在机器人任务中取得了显著进展。本文利用基础模型来解决抓取检测这一机器人学中具有广泛工业应用的长久挑战。尽管已有大量抓取数据集，但与真实世界相比，其物体多样性仍然有限。幸运的是，基础模型拥有丰富的真实世界知识库，涵盖我们日常生活中的各种物体。因此，解决先前抓取数据集表征不足的一个有前景的方案，是挖掘这些基础模型中蕴含的通用知识。我们提出了 Grasp-Anything，一个利用基础模型合成的新型大规模抓取数据集。Grasp-Anything 在多样性和规模上表现卓越，拥有 100 万样本及文本描述，超过 300 万个物体，远超以往数据集。实验表明，Grasp-Anything 成功地在视觉任务和真实机器人实验中支持零样本抓取检测。

### 核心要点提炼
- **研究背景**：现有抓取数据集物体类别少、场景受限、缺乏语言描述，且多为 model-centric 方法，跨域泛化差
- **研究动机**：利用基础模型的通用世界知识，生成一个能覆盖真实世界中几乎无限物体和场景的大规模抓取数据集
- **核心方法**：ChatGPT 提示工程生成场景描述 → Stable Diffusion 合成图像 → OFA+SAM 实例分割 → 预训练模型 + 分析评估标注抓取姿态 → 整合为 1M 样本数据集
- **主要结果**：在三个抓取网络上实现 zero-shot 抓取检测显著提升；KUKA 机器人实验 93.3%（单物体）和 91.6%（杂乱场景）；跨数据集迁移提升 9-29%
- **研究意义**：首个 data-centric 大规模抓取数据集，利用基础模型生成训练数据的新范式

## 研究背景与动机

### 领域现状
抓取检测（Grasp Detection）是机器人学的基本问题。近年来深度学习推动了许多抓取方法的进步，但这些工作主要是 **model-centric**（以模型为中心）——专注于改进神经网络架构，而数据集的改进相对滞后。

### 现有方法的局限性

1. **物体多样性不足**：现有数据集的物体类别严重受限。以物体数量最多的 VMRD 为例也仅约 15K 个物体，而 MetaGraspNet 作为最新点云数据集只有 82 个物体
2. **场景单一**：多数数据集假设 bin-like（箱式）配置或实验室控制环境，与真实世界的自然场景差距巨大
3. **缺乏语言描述**：几乎所有抓取数据集都不含自然语言描述，限制了人机交互和语言驱动抓取的发展
4. **标注成本高**：许多数据集依赖昂贵的试错（trial-and-error）或众包标注，存在偏差

### 研究动机
基础模型（ChatGPT 等）拥有对真实世界的泛知识表达，理论上可以描述出近乎无限的物体和场景。如果能利用这些知识自动合成大规模、多样化的抓取数据集，就能从根本上解决 data-centric 的泛化问题。

## 研究问题

### 核心研究问题
- 能否利用基础模型（ChatGPT + Stable Diffusion）合成一个覆盖真实世界物体多样性的抓取数据集？
- 在该合成数据集上训练的抓取网络能否实现 zero-shot 泛化？
- 合成数据能否直接迁移到真实机器人抓取任务？

## 方法概述

### 核心思想
用一个 Data-Centric 的方法替代传统的 Model-Centric 方法：不再改进网络架构，而是用基础模型的通用知识自动"造"一个无比多样化的抓取数据集。

### 方法框架

**三阶段 Pipeline**：

![[generation_pipeline.pdf|800]]

> 图1：Grasp-Anything 数据集生成流程。阶段 1：ChatGPT 提示工程生成场景描述；阶段 2：Stable Diffusion 合成图像 + OFA/SAM 实例分割；阶段 3：抓取姿态自动标注与分析评估

#### 阶段 1：场景生成（Scene Generation）

**目标**：利用 ChatGPT 生成 100 万条多样化的场景描述文本

**(i) 指令初始化（Directives Initialization）**

通过精心设计的对话配置 ChatGPT 的目标和行为：
```
Q: "Imagine you are helping me to generate a corpus of scene descriptions,
   each condensed to a single sentence. My goal is to generate as many
   diverse graspable objects as possible. Each sentence must be distinct
   and should contain at least two objects."

Q: "The template for each sentence contains two parts. The first part is
   the sentence... The second part is the list of extracted objects..."
```

模板约束确保每条输出包含：
- `<Obj_1> <Obj_2> ... <Verb> <Container>` 格式的场景描述
- `[Obj_1, Obj_2, ...]` 格式的可抓取物体列表

**(ii) 上下文增强（Context Augmentation）**

创建一个自我强化循环来解决 ChatGPT 长期生成中的幻觉问题：
1. 初始化 Prompt Buffer，手动填充前 50 个样本
2. 每次从 Buffer 随机抽取 10-15 条作为 in-context examples 喂入 ChatGPT
3. ChatGPT 生成新描述后追加回 Buffer
4. 循环迭代直至 1M 条场景描述

这个过程保证长时间范围内生成质量和多样性的稳定性。

#### 阶段 2：图像合成（Image Synthesis）

- **Stable Diffusion 2.1**：将场景描述文本转换为对应图像
- **OFA**（视觉定位）：在图像中定位每个物体
- **SAM**（Segment-Anything）：为每个物体生成实例分割掩码

最终得到：图文 + 每个物体的实例掩码

#### 阶段 3：抓取姿态标注与评估（Grasp Pose Annotation）

- **标注**：使用预训练抓取模型 RAGT-3/3 在物体掩码上生成候选抓取姿态（矩形表示法）
- **评估与过滤**：使用 Kamon et al. 的分析性方法评估抓取质量

![[convex_hull.pdf|300]] ![[grasp_evaluation.pdf|300]]

> 图2：左：对物体分割掩码构建凸包；右：抓取姿态的扭矩分析

**核心公式 —— 抓取质量评分**：

抓取质量由净扭矩决定：

$$\mathcal{T} = \underbrace{(\tau_1+\tau_2)}_{\text{Resistance}} - \underbrace{RMg}_{\text{Torque}}$$

其中 $\tau_i = K\mu_sF\cos\alpha_i$，$i\in\{1, 2\}$

由于物理参数 $M, K, \mu_s$ 在实际中难以获取，论文推导了一个无需物理参数的简化评分：

$$\tilde{\mathcal{T}} = \frac{\cos\alpha_1 + \cos\alpha_2}{R}$$

- $\tilde{\mathcal{T}} > 0$ → 正抓取（positive grasp）
- $\tilde{\mathcal{T}} \leq 0$ → 负抓取（negative grasp）
- 对数抓取（antipodal grasp）通常产生更大的正 $\tilde{\mathcal{T}}$

### 数据集统计

| 指标 | 数值 |
|------|------|
| 总样本数 | **1,000,000+** |
| 总物体数 | **3,000,000+** |
| LVIS 物体类别 | **236 类** |
| POS 标签 | **1.5M** (35% 名词, 20% 形容词, 7% 动词) |
| 数据格式 | 图像 + 文本描述 + 抓取矩形 + 实例掩码 |
| 生成耗时 | 约 3 个月 (3× NVIDIA Quadro RTX 8000) |

![[num_cats.pdf|700]]

> 图3：按 LVIS 类别统计 —— Grasp-Anything 覆盖 236 个 LVIS 类别，远超其他数据集

![[number_of_objects.pdf|350]]

> 图4：物体数量对数比较 —— Grasp-Anything 拥有 ~3M 物体，相比第二名 VMRD（~15K）高出两个数量级

![[pos-tags.pdf|350]]

> 图5：POS 标签分布 —— 包含丰富的词汇多样性，35% 名词 + 20% 形容词 + 7% 动词

![[shape_visualization.pdf|700]]

> 图6：物体形状热力图 —— Grasp-Anything（左）vs Jacquard（右），Grasp-Anything 覆盖更大图像区域，形状多样性更高

### 数据集样本

![[grasp_visualization.pdf|800]]

> 图7：Grasp-Anything 数据集样本 —— 每个物体显示评分最高的抓取姿态（粗线标记），场景排列自然多样化

## 实验结果

### 实验目标
验证两个核心问题：
1. Grasp-Anything 能否作为抓取检测的挑战性基准？
2. 合成数据集能否迁移到真实机器人？

### 实验设置

#### 基线网络
- **GR-ConvNet**：基于 ResNet 的抓取检测网络
- **Det-Seg-Refine**：检测-分割-精炼三步法
- **GG-CNN**：轻量级生成式抓取网络

#### 对比数据集
- Jacquard（54K 样本，模拟）
- Cornell（1035 样本，真实）
- VMRD（4683 样本，真实）
- OCID-grasp（11K 样本，真实）

#### 评估指标
- **Success Rate**：IoU > 0.25 且角度偏差 < 30° 的抓取占比
- **Harmonic Mean (H)**：衡量 Base 和 New 类别上的综合表现

### 主要结果

#### 1. Base-to-New 泛化（数据集内）

| Baseline | 数据集 | Base | New | H |
|----------|--------|------|-----|---|
| GR-ConvNet | **Grasp-Anything** | 0.74 | 0.61 | **0.67** |
| GR-ConvNet | Jacquard | 0.88 | 0.66 | 0.75 |
| GR-ConvNet | Cornell | 0.98 | 0.74 | 0.84 |
| Det-Seg-Refine | **Grasp-Anything** | 0.64 | 0.59 | 0.61 |
| GG-CNN | **Grasp-Anything** | 0.71 | 0.59 | 0.64 |

> Grasp-Anything 的得分较低（Base 0.74 vs Cornell 0.98）恰恰说明其难度更高——因为它覆盖了更多未见过的测试物体

#### 2. 跨数据集迁移（核心亮点）

使用 GR-ConvNet，训练/测试在不同数据集：

| 训练集 ↓ / 测试集 → | Jacquard | Cornell | VMRD | OCID | Grasp-Anything |
|---------------------|----------|---------|------|------|----------------|
| Jacquard | 0.89 | 0.51 | 0.13 | 0.21 | 0.17 |
| Cornell | 0.06 | 0.98 | 0.20 | 0.12 | 0.15 |
| VMRD | 0.06 | 0.21 | 0.79 | 0.11 | 0.12 |
| OCID-grasp | 0.09 | 0.12 | 0.20 | 0.74 | 0.12 |
| **Grasp-Anything** | **0.38** | **0.60** | **0.32** | **0.37** | 0.67 |

> 关键发现：在 Grasp-Anything 上训练的模型迁移到 Jacquard 上测试，准确率为 0.38，是其他数据集训练结果（0.06-0.09）的 **4 倍以上**

#### 3. 泛化零样本抓取检测

在 Cornell、VMRD、OCID-grasp 上测试，Grasp-Anything 训练 vs Jacquard 训练：

| Baseline | 测试集 | Train=Jacquard (H) | Train=Grasp-Anything (H) | 提升 |
|----------|--------|---------------------|--------------------------|------|
| GR-ConvNet | Cornell | 0.51 | **0.62** | +21% |
| GR-ConvNet | VMRD | 0.13 | **0.31** | +138% |
| GR-ConvNet | OCID | 0.21 | **0.39** | +86% |

#### 4. 真实机器人实验（KUKA 机器人）

![[robotic-experiment.pdf|350]]

> 图8：KUKA 机器人实验设置，使用 RealSense 深度相机

| 训练数据集 | 单物体 | 杂乱场景 |
|-----------|--------|----------|
| Cornell | 81.6% | 68.3% |
| Jacquard | 85.0% | 88.3% |
| VMRD | 78.3% | 70.0% |
| OCID-grasp | 80.0% | 71.6% |
| **Grasp-Anything** | **93.3%** | **91.6%** |

> 每个场景测试 60 次，GR-ConvNet 网络，15 个物体

![[qualitative_result.pdf|800]]

> 图9：定性对比 —— 同一办公室场景图像上，GR-ConvNet 在不同数据集训练后的抓取检测结果。Grasp-Anything 训练的模型检测到更多且更合理的抓取姿态

![[in_the_wild.png|800]]

> 图10：In-the-wild 泛化 —— Grasp-Anything 训练的模型在 NBMOD、YCB-Video、GraspNet 及网络图片上的零样本检测效果

### 消融与分析

论文没有传统意义上的消融实验（因为不是方法论工作），而是通过以下方式做分析：

1. **物体形状分布比较**：Grasp-Anything vs Jacquard 的热力图覆盖面积对比，证明多样性
2. **POS 标签统计**：证明语言描述的丰富性
3. **LVIS 类别覆盖**：236 vs 115（VMRD），证明类别广度

## 深度分析

### 研究价值评估

#### 理论贡献
- **Data-Centric 范式验证**：首次系统验证了"改进数据比改进模型更重要"在抓取检测中的有效性——使用同一个 GR-ConvNet，仅换数据集就能在跨域迁移中提升 4 倍
- **基础模型用于数据合成**：证明了 ChatGPT + Stable Diffusion 的组合可以自动构建具有真实泛化价值的数据集，而非简单的数据增强
- **无物理参数的抓取评估**：将 Kamon et al. 的分析性方法简化为仅需几何变量（$\cos\alpha_1 + \cos\alpha_2)/R$），在不依赖物理参数的情况下进行抓取筛选

#### 实际应用价值
- **零样本抓取**：在 Grasp-Anything 上训练的网络可以直接部署到未见过的物体和场景，无需重新标注
- **跨域迁移**：作为一个 universal dataset，可以显著提升其他小规模真实数据集的训练效果
- **语言驱动抓取基础**：100 万条场景文本描述为语言引导的抓取交互奠定了基础

#### 领域影响
- **短期**：为抓取检测社区提供一个有挑战性的大规模基准
- **中期**：推动 data-centric 思维在机器人数据集中普及
- **长期**：基础模型合成训练数据有望成为机器人学习的标准范式

### 方法优势详解

#### 优势 1：数据规模碾压
- 3M 物体 vs VMRD 15K，两个数量级的领先
- 236 LVIS 类别 vs 第二名 115
- 100 万样本，80% 以上用于训练（其他数据集通常几千到几万）

#### 优势 2：自然场景覆盖
- 大多数数据集是 lab-controlled 或 bin-like 场景
- ChatGPT 生成的场景描述天然包含"自然生活场景"（如"a cup and a book on a desk"）

#### 优势 3：多模态标注
- 同时包含：图像 + 场景文本 + 物体列表 + 抓取矩形 + 实例掩码
- 这是第一个同时具备文本描述和抓取标注的大规模数据集

### 局限性分析

#### 局限 1：缺乏 3D / 深度信息
- 所有数据为 2D 图像 + 矩形抓取表示，不支持 6-DoF 抓取
- 真实机器人实验中仍需依赖 RealSense 深度相机来估计深度
- 原因：text/image-to-point-cloud 基础模型尚未成熟
- 可能缓解：未来可以用扩散模型生成点云或多视图重建

#### 局限 2：生成耗时且依赖商业 API
- 生成 1M 样本需要 3 台 RTX 8000 跑 3 个月
- 依赖 ChatGPT 商业 API（2023 年的 GPT-3.5/4）
- 影响：其他研究者难以重现数据生成过程（但论文提供了已生成的数据）

#### 局限 3：抓取标注质量依赖预训练模型
- RAGT-3/3 作为自动标注器本身有误差
- 分析性评估（$\tilde{\mathcal{T}}$）只提供启发式过滤，不能完全替代物理验证
- 可能在边缘情况下产生低质量标注

#### 局限 4：合成域间隙（Sim-to-Real Gap）
- Stable Diffusion 生成的图像与真实相机图像存在分布差异
- 在真实数据集测试时（Table 4 跨数据集），准确率仍远低于同域训练（0.38 vs 0.89 on Jacquard）

### 适用性与场景分析

#### 适用场景
- **预训练 + 微调**：在 Grasp-Anything 上预训练，再在目标域（如特定工厂场景）微调
- **零样本快速部署**：直接部署到未见过的日常物体抓取
- **语言引导抓取研究**：利用数据集中的文本描述训练 language-conditioned grasp 模型

#### 不适用场景
- **高精度 6-DoF 抓取**：只有矩形表示，不支持 6-DoF 抓取
- **接触约束严格的工业任务**：如精密装配、力控操作
- **透明/反光物体**：Stable Diffusion 生成的此类物体图像可能不够真实

## 与相关论文对比

### 对比论文选择依据
选择与本文同属抓取数据集或数据生成范式的代表性工作

| 对比维度 | Jacquard | MetaGraspNet | **Grasp-Anything（本文）** |
|----------|----------|--------------|---------------------------|
| 数据来源 | 模拟（PyBullet） | 模拟（Isaac Sim） | ChatGPT + Stable Diffusion 合成 |
| 物体规模 | 11K 物体 | 82 物体 | **3M 物体** |
| 类别数 | -- | 82 | **236（LVIS）** |
| 抓取表示 | 矩形 | 6-DoF | 矩形 |
| 语言描述 | 无 | 无 | **100 万文本** |
| 场景自然度 | 简单摆放 | 物理模拟摆放 | **日常自然场景描述** |
| 真实机器人验证 | 有限 | 有 | **93.3% / 91.6%** |
| 年份 | 2018 | 2022 | 2023 |

## 技术路线定位

### 所属技术路线
本文属于 **Data-Centric Robotics** + **Foundation Models for Data Generation** 路线：

```
传统数据收集（手工/模拟）
  → 数据增强（裁剪/旋转/颜色抖动）
    → 生成模型数据合成（GAN-based）
      → [本文] 基础模型大规模合成（ChatGPT + Stable Diffusion）
        → 未来: 端到端 3D 数据合成（text-to-point-cloud）
```

### 本文在技术路线中的位置
- **承上**：继承了抓取检测领域矩形表示的传统，沿用了 GR-ConvNet 等成熟网络
- **启下**：开启了利用 LLM/T2I 模型生成机器人训练数据的新方向
- **关键节点**：证明了"合成数据可以超越真实数据"在特定场景下的可行性

### 相关论文关系
- 被后续工作引用：作为数据合成范式的先驱之一
- 与 [[VLA_Embodied]] 中的多篇语言驱动操纵论文形成互补——它提供了数据基础

## 未来工作建议

### 作者建议
1. **扩展到 3D/点云**：等待 text-to-point-cloud 基础模型成熟后补充 3D 数据
2. **语言驱动抓取**：利用 100 万文本描述训练 language-conditioned 抓取网络
3. **sim2real 迁移**：将 Grasp-Anything 训练的模型迁移到更复杂的真实场景

### 基于分析的未来方向
1. **多模态基础模型联合合成**：用单一 unified model（如 GPT-4V）同时生成场景描述、图像和抓取标注
2. **自适应难度采样**：让生成的场景描述从简单到复杂逐渐过渡，构建课程学习数据
3. **与 3DGS 结合**：用 3D Gaussian Splatting 重建生成的场景，为每个高斯球附加 affordance 标签
4. **封闭循环验证**：生成 → 抓取 → 物理模拟验证 → 反馈至生成模型 → 改善数据质量

## 我的综合评价

### 价值评分

#### 总体评分
**7.5/10** — ICRA 2024 论文，想法清晰、执行扎实、实验全面。核心亮点是用基础模型造数据的范式创新，但方法本身（提示工程 + SD + 现成分割模型）技术深度不高，更像是"工程整合"。数据集本身价值很大。

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | 范式创新（data-centric + foundation model synthesis），但具体技术组件均为现有方法 |
| 技术质量 | 7/10 | 流水线设计合理，无物理参数的抓取评估公式巧妙，但管道本身是黑盒组合 |
| 实验充分性 | 9/10 | 5 个数据集 × 3 个网络 × 跨数据集 + 真实机器人，非常全面 |
| 写作质量 | 8/10 | 清晰易懂，图表丰富，对比全面 |
| 实用性 | 8/10 | 数据集和代码公开，可直接用于抓取研究预训练 |

### 重点关注
- 提示工程的自我强化循环设计（Context Augmentation）——用 prompt buffer 维持长期生成的多样性和质量
- 无物理参数的抓取质量评估公式 $(\cos\alpha_1 + \cos\alpha_2)/R$
- 跨数据集实验的结论：Grasp-Anything 训练 → Jacquard 测试是反向的 4x+

## 我的笔记

%% 这篇论文最值得关注的是它代表了机器人学习中一个重要的范式转变——从"设计更好的模型"转向"创造更好的数据"。2023-2024 年这种思路在机器人社区越来越流行，Grasp-Anything 是这个趋势的先行者之一。

在 VLA 研究中，大量高质量训练数据的重要性已被反复验证（如 RT-2、Octo 等），但如何高效、低成本地获取这些数据仍是一个开放问题。Grasp-Anything 提供了一条思路：用 LLM 的知识来合成数据。

不过一个值得思考的问题是：ChatGPT 的知识是有偏的（倾向常见物体和典型场景），这是否会引入新的偏置？论文没有讨论这一点。 %%

## 相关论文

### 直接相关
- RAGT-3/3 (Cao et al. 2023) — 用于抓取标注的预训练模型

### 背景相关
- Jacquard (Depierre et al. 2018) — 关键对比数据集
- MetaGraspNet (Gilles et al. 2022) — 最新点云抓取数据集
- OFA (Wang et al. 2022) — 视觉定位模型
- SAM (Kirillov et al. 2023) — 分割一切模型

### 后续工作
- AnyGrasp (Fang et al. 2023) — 通用抓取检测
- 语言驱动抓取方向（Xu et al. 2023, Yang et al. 2023）

## 外部资源
- 项目页面：[https://airvlab.github.io/grasp-anything/](https://airvlab.github.io/grasp-anything/)
- 代码 & 数据集：[https://github.com/airvlab/grasp-anything](https://github.com/airvlab/grasp-anything)

> [!tip] 关键启示
> 改进数据（data-centric）可能比改进模型（model-centric）更能提升抓取泛化能力——同一个网络在更大的数据集上训练，迁移能力可提升 4 倍以上。

> [!warning] 注意事项
> - 数据集是 2D 矩形抓取，不支持 6-DoF，实际机器人使用需额外深度估计
> - 合成图像与真实相机图像存在分布差异，直接部署可能有 sim-to-real gap
> - 生成需要 ChatGPT API，数据生成过程难以复现（但预生成数据公开）

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读！如果你是做抓取/操纵/VLA数据的研究者，这篇论文的方法和数据都值得了解，尤其是它展示的 data-centric 思维转变。
