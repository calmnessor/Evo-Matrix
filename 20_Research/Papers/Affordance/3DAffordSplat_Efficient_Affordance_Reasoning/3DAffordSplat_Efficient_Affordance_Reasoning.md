---
date: "2026-05-12"
paper_id: "arXiv:2504.11218"
title: "3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians"
authors: "Zeming Wei, Junyi Lin, Yang Liu, Weixing Chen, Jingzhou Luo, Guanbin Li, Liang Lin"
domain: "Affordance"
tags:
  - 论文笔记
  - Affordance
  - 3DGS
  - 3D-Gaussian-Splatting
  - Cross-Modal
  - Point-Cloud
  - Dataset
  - CVPR-2025
  - Embodied-AI
  - Affordance-Reasoning
  - Multi-Modal
quality_score: "8.5/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# 3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians

## 核心信息
- **论文ID**：arXiv:2504.11218
- **作者**：Zeming Wei, Junyi Lin, Yang Liu, Weixing Chen, Jingzhou Luo, Guanbin Li, Liang Lin（中山大学）
- **机构**：Sun Yat-sen University, Peng Cheng Laboratory, Guangdong Key Laboratory of Big Data Analysis and Processing
- **发布时间**：2025-04-15
- **会议/期刊**：CVPR 2025
- **链接**：[arXiv](https://arxiv.org/abs/2504.11218) | [GitHub](https://github.com/HCPLab-SYSU/3DAffordSplat)
- **引用**：--

## 摘要翻译

### 英文摘要
3D affordance reasoning is essential in associating human instructions with the functional regions of 3D objects, facilitating precise, task-oriented manipulations in embodied AI. However, current methods, which predominantly depend on sparse 3D point clouds, exhibit limited generalizability and robustness due to their sensitivity to coordinate variations and the inherent sparsity of the data. By contrast, 3D Gaussian Splatting (3DGS) delivers high-fidelity, real-time rendering with minimal computational overhead by representing scenes as dense, continuous distributions. This positions 3DGS as a highly effective approach for capturing fine-grained affordance details and improving recognition accuracy. Nevertheless, its full potential remains largely untapped due to the absence of large-scale, 3DGS-specific affordance datasets. To overcome these limitations, we present 3DAffordSplat, the first large-scale, multi-modal dataset tailored for 3DGS-based affordance reasoning. This dataset includes 23,677 Gaussian instances, 8,354 point cloud instances, and 6,631 manually annotated affordance labels, encompassing 21 object categories and 18 affordance types. Building upon this dataset, we introduce AffordSplatNet, a novel model specifically designed for affordance reasoning using 3DGS representations. AffordSplatNet features an innovative cross-modal structure alignment module that exploits structural consistency priors to align 3D point cloud and 3DGS representations, resulting in enhanced affordance recognition accuracy. Extensive experiments demonstrate that the 3DAffordSplat dataset significantly advances affordance learning within the 3DGS domain, while AffordSplatNet consistently outperforms existing methods across both seen and unseen settings, highlighting its robust generalization capabilities.

### 中文翻译
3D affordance reasoning 是将人类指令与 3D 物体功能区域关联的核心能力，支撑着具身 AI 中的精确任务导向操作。然而，当前方法主要依赖稀疏 3D 点云，由于对坐标变化的敏感性和数据固有的稀疏性，泛化性和鲁棒性受限。相比之下，3D Gaussian Splatting（3DGS）通过将场景表示为密集、连续分布，以最小的计算开销实现了高保真实时渲染。这使 3DGS 成为捕获细粒度 affordance 细节和提升识别精度的有效方法。但其潜力因缺乏大规模 3DGS affordance 数据集而未被充分挖掘。为此，我们提出 3DAffordSplat——首个大规模、多模态 3DGS affordance reasoning 数据集，包含 23,677 个 Gaussian 实例、8,354 个点云实例和 6,631 个人工标注的 affordance 标签，覆盖 21 个物体类别和 18 种 affordance 类型。基于此数据集，我们提出 AffordSplatNet，专为 3DGS 表示设计的 affordance reasoning 模型，核心创新的跨模态结构对齐模块利用结构一致性先验对齐点云和 3DGS 表示，显著提升识别精度。

### 核心要点提炼
- **研究背景**：3D affordance 是具身 AI 中连接感知与动作的关键能力，但现有方法主要基于稀疏点云，缺乏大规模 3DGS affordance 数据集
- **研究动机**：3DGS 的密集连续表示天然适合 affordance reasoning，但缺乏专门的数据集和模型
- **核心方法**：首个 3DGS affordance 数据集 + AffordSplatNet 模型（PointNet++ 骨干 + RoBERTa 语言编码 + 跨模态结构对齐 + 粒度自适应选择 + 动态核解码）
- **主要结果**：Seen mIoU=33.03/AUC=84.67，Unseen mIoU=18.91，显著超越 PointRefer 和 IAGNet
- **研究意义**：开创了 3DGS 在 affordance reasoning 领域的应用，提供了数据和模型双重基准

## 研究背景与动机

### 领域现状
Affordance reasoning 是具身 AI 的核心能力——识别 3D 物体的"可操作区域"（如可抓握、可推、可旋转的部件），使机器人能够根据人类指令执行精确操作。

现有方法主要基于三种模态：
1. **2D 图像**：缺乏深度信息，无法捕获完整 3D 结构，遮挡时表现差
2. **视频**：不提供直接 3D 空间信息，难以标注细微动态变化
3. **点云**：虽提供直接 3D 几何表示，但本质上是离散采样，无法表示连续曲面和精细几何特征

### 现有方法的局限性
- 点云的稀疏性和坐标敏感性导致 affordance 识别精度受限
- 缺乏大规模 3DGS affordance 数据集（现有 3DGS 数据集如 CLIP-GS、ShapeSplat 均无 affordance 标注）
- 现有 affordance 模型设计为处理离散数据（点云/图像），无法利用 3DGS 的连续特性
- 传统 3DGS 语义嵌入方法（参数化扩展静态分配单一语义特征）无法处理多属性 affordance 场景

### 研究动机
3DGS 的三大特性使其天然适合 affordance reasoning：
1. 高几何精度和表面细节保留——解决点云的离散性和不完整性
2. 丰富的颜色信息集成——弥补图像方法 3D 空间信息缺失
3. 高效实时渲染（1080p 30+ fps）——低计算开销

但"没有大规模标注数据集"和"没有适配 3DGS 的模型架构"是两大核心障碍。

## 研究问题

### 核心研究问题
1. 如何构建首个大规模多模态 3DGS affordance 数据集？
2. 如何设计适配 3DGS 连续特性的 affordance reasoning 模型？
3. 如何利用点云 affordance 数据辅助 3DGS affordance 学习？

## 方法概述

### 核心思想
**"数据集 + 模型"双重贡献**：通过整合 ShapeSplat（3DGS 数据）和 LASO（点云+文本数据），构建首个多模态 3DGS affordance 数据集。在此基础上，设计 AffordSplatNet——利用跨模态结构对齐（CMSA）将点云 affordance 知识迁移到 3DGS 域，通过粒度自适应选择和多模态融合实现精确 affordance 预测。

### 方法框架

#### 整体架构

![[Model-new.pdf|800]]

> 图1：AffordSplatNet 整体架构。(a) 分层处理流水线：3D Gaussian 经 PointNet++ 提取多粒度特征，RoBERTa 从文本查询推理 ⟨Aff⟩ token，通过交叉注意力和通道注意力融合，粒度自适应门控选择最优尺度，动态核解码生成 affordance mask。(b) CMSA 模块：在预训练阶段对齐 Gaussian 和点云的 affordance 区域结构关系。

#### 3DAffordSplat 数据集

![[Datasetpipeline.pdf|800]]

> 图2：数据集构建流程。整合 LASO（点云+文本）和 ShapeSplat（3DGS），覆盖 21 类别、18 种 affordance 类型。

| 属性 | 数值 |
|------|------|
| Gaussian 实例 | 23,677 |
| 点云实例 | 8,354 |
| 人工标注 affordance | 6,631 |
| 物体类别 | 21 |
| Affordance 类型 | 18 |
| 问答题模板 | 15 个/Object-Affordance 对 |

**Seen/Unseen 设置**：
- **Seen**：训练和测试共享相同的物体类别和 affordance 类型分布
- **Unseen**：测试集包含全新的物体类别、affordance 类型和 Object-Affordance 组合

#### 各模块详细说明

**特征编码**

- **3D Gaussian 编码**：仅使用结构特征 $\boldsymbol{\mathcal{G}}_{\text{struct}} = \{\boldsymbol{m, s, r}\} \in \mathbb{R}^{10}$（中心位置 + 尺度 + 旋转），通过 PointNet++ 提取三级多粒度几何特征 $\{\boldsymbol{F_{g}^{i}}\}^3_{i=1}$
- **文本编码**：RoBERTa 预训练语言模型，特别设计的 $\langle \text{Aff} \rangle$ token 捕获 affordance 中间表示
- **关键设计选择**：仅用结构参数（xyz+rotation+scale），消融验证 mIoU 最优（51.20），优于加入 opacity+RGB 的组合

**多模态融合**
- 空间级融合：以 $\boldsymbol{H}_{Aff}$ 为 query，几何特征为 key/value 进行交叉注意力
- 通道级融合：通道注意力机制自适应重标定跨模态特征
- 残差连接保持几何信息保真度

**粒度自适应选择（Granularity-Adaptive Selection）**
- IDW 插值将多粒度特征统一到相同分辨率
- 可学习门控权重 Softmax 竞争分配跨粒度重要性：
  $$\boldsymbol{W} = \operatorname{Softmax} (\boldsymbol{W}_{gate}\odot\left[{\boldsymbol{\overline{F}}}_1 \| {\boldsymbol{\overline{F}}}_2 \|{\boldsymbol{\overline{F}}}_3\right])$$
- 确保不同任务自适应选择最优空间尺度

**动态核解码器**
- IDW 上采样到原始 Gaussian 密度
- 有效性 mask 过滤 padding 位置
- Transformer Decoder 生成位置感知动态核
- 卷积生成最终 affordance mask：
  $$\mathcal{M}_{gs} = \sigma (\boldsymbol{F}_{\text{valid}} \ast \boldsymbol{K}_{dynamic}) \odot \boldsymbol{M}_{\text{valid}}$$

**跨模态结构对齐（CMSA）**
- 核心思想：同一物体类别，虽然 3DGS 和点云的显式表示不同，但 affordance 区域与整体结构的相对空间关系保持不变
- 将 Gaussian affordance 区域和点云 affordance 区域编码到共享空间
- 通过共享多头交叉注意力计算结构亲和矩阵
- Chamfer Distance 计算跨模态结构相似度作为损失权重：
  $${w}_{consis}^i = \operatorname{Softmax}(-\mathcal{D}_{Chamfer}(\boldsymbol{\mathcal{G}}_{\text{struct}},\boldsymbol{\mathcal{P}}_k) / \tau)$$
- 余弦损失对齐跨模态相对结构关系

### 训练策略

**两阶段训练**：

| 阶段 | 目标 | 损失函数 | 数据 | Epoch |
|------|------|----------|------|-------|
| Pretrain | 跨模态结构对齐 | $\mathcal{L}_{pretrain} = \mathcal{L}_{consis} = w_{consis} \odot \mathcal{L}_{cosine}$ | 94,708 GS-PC 对（无标注） | 1 |
| Finetune | 精确 affordance 预测 | $\mathcal{L}_{finetune} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice} + \mathcal{L}_{text}$ | 6,631 标注 GS | 60 |

- 预训练学习率 1e-5，微调 1e-4
- LoRA 微调语言模块，全微调其他组件
- 4× RTX 4090 GPU

## 实验结果

### 实验设置

#### 基线方法
- **PointRefer**：SOTA 语言-点云 affordance 模型
- **IAGNet**：SOTA 图像-点云 affordance 模型

#### 评估指标
- **mIoU**：预测区域与 GT 区域的重叠度，核心指标
- **AUC**：ROC 曲线下面积，衡量 saliency map 质量
- **SIM**：预测图与 GT 图的分布相似度
- **MAE**：平均绝对误差

### 主要结果

#### 数据集验证

![[Experiment1.pdf|400]]

> 图3：点云（稀疏、离散）vs 3DGS（密集、连续、纹理丰富）在 affordance 表示上的对比。

| 设置 | 方法 | 训练/验证集 | 测试集 | 微调 | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|------|------|------------|--------|------|-------|------|------|------|
| Seen | PointRefer | LASO | 3DAffordSplat | ✗ | 5.10 | 52.10 | 0.17 | 0.26 |
| Seen | PointRefer | LASO | 3DAffordSplat | ✓ | 49.40 | 93.60 | 0.61 | 0.12 |
| Seen | PointRefer | 3DAffordSplat | 3DAffordSplat | -- | **51.70** | **94.00** | **0.63** | **0.11** |
| Unseen | PointRefer | 3DAffordSplat | 3DAffordSplat | -- | 18.30 | 66.50 | 0.28 | 0.28 |

**关键发现**：
1. **高质量标注价值**：3DAffordSplat 上的最佳结果（mIoU 51.70）远超 LASO 原生训练（19.20），验证了标注质量优势
2. **pc→gs 迁移更优**：点云→3DGS 微调后性能恢复显著（5.10→49.40），而反向迁移有限（3.80→18.50）
3. **微调必要性**：不微调时跨模态迁移性能骤降，证明需要专门的 3DGS affordance 数据集

#### 模型对比

| 设置 | 方法 | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|------|------|-------|------|------|------|
| Seen | IAGNet | 14.63 | 56.67 | 0.35 | 0.41 |
| Seen | PointRefer | 18.40 | 78.50 | 0.43 | 0.20 |
| Seen | **AffordSplatNet** | **33.03** | **84.67** | **0.46** | **0.21** |
| Unseen | IAGNet | 4.70 | 40.77 | 0.24 | 0.43 |
| Unseen | PointRefer | 15.90 | 67.00 | 0.31 | 0.29 |
| Unseen | **AffordSplatNet** | **18.91** | **66.71** | **0.32** | **0.31** |

### 消融实验

#### 语言编码器消融

| 语言编码器 | 类型 | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|-----------|------|-------|------|------|------|
| BART | Encoder-Decoder | 20.61 | 73.52 | 0.35 | 0.27 |
| GPT-2 | Decoder-only | 32.96 | 81.34 | 0.44 | 0.22 |
| **RoBERTa** | **Encoder-only** | **33.03** | **84.67** | **0.46** | **0.21** |

RoBERTa 最优，归因于双向上下文编码和对 MLLM 的适配性。

#### CMSA 消融

| 设置 | 变体 | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|------|------|-------|------|------|------|
| Seen | Ours (Pretrain+Finetune) | 33.03 | 84.67 | 0.46 | 0.21 |
| Seen | w/o CMSA (仅 Finetune) | **37.18** | 81.34 | 0.48 | 0.20 |
| Unseen | Ours (Pretrain+Finetune) | **18.91** | **66.71** | **0.32** | 0.31 |
| Unseen | w/o CMSA (仅 Finetune) | 17.93 | 62.39 | 0.27 | 0.31 |

**CMSA 的双面性**：
- **Unseen 提升显著**（AUC: 66.71 vs 62.39）：CMSA 将点云 affordance 先验迁移到 3DGS，对未见物体泛化至关重要
- **Seen 反而降低**（mIoU: 33.03 vs 37.18）：**Task-Specific Overalignment 现象**——预训练对齐可能施加过强的特征对应约束，与微调数据产生冲突；对于数据充足的 Seen 场景，预训练收益被微调数据覆盖

### 定性结果

![[show-cases-row-1.pdf|800]]

> 图4：AffordSplatNet 可视化结果。模型能精确分割细粒度 affordance（如 Door-Open），同时捕获大范围连续区域（如 Clock-Display）。

![[Real-world-cases.pdf|400]]

> 图5：真实世界案例。使用 3DGS 重建真实物体，模型成功识别 Mug-Grasp 和 Bag-Grasp 区域。

**失败案例分析**：主要源于 (1) 轻量语言模型词汇覆盖不足导致错误回答，(2) 对多不连续 affordance 区域的复杂物体处理不佳。

## 深度分析

### 研究价值评估

#### 理论贡献
- **首个 3DGS affordance 数据集**：填补了 3DGS + affordance 交叉领域的空白
  - 创新点：首次将 3DGS、点云、文本三种模态以 affordance 标注对齐
  - 学术价值：为后续 3DGS affordance 研究提供了标准化基准
- **跨模态结构对齐机制**：从"结构一致性先验"出发对齐点云和 3DGS 表示
  - 创新点：利用 Chamfer Distance 加权的余弦损失实现无监督跨模态对齐
  - 发现：揭示了预训练-微调范式中的 Task-Specific Overalignment 现象

#### 实际应用价值
- **机器人任务规划**：结合 LLM planner 实现新物体的零样本 affordance 推理
- **AR 交互界面**：实时渲染 + affordance 推理 → 家具布局辅助、维护培训系统
- **智能家居**：语音激活系统识别"可推开的柜门"、"可提起的沙发垫"
- **工业质检**：对比理想 Gaussian affordance map 与产线 LiDAR 扫描，检测功能缺陷

### 方法优势详解

#### 优势1：3DGS 的连续表示
- 相比点云离散采样，3DGS 的密集连续分布能更好捕获曲面 affordance
- 丰富的纹理信息提供了点云缺失的外观线索
- 实验证据：点云→3DGS 迁移优于 3DGS→点云

#### 优势2：粒度自适应选择
- 不同 affordance 需要不同空间粒度（如"抓握"需要精细局部，"坐"需要整体表面）
- 通过竞争性门控动态选择最优粒度
- 实验证据：在 Door-Open（精细）和 Clock-Display（大面积）上均表现优异

#### 优势3：两阶段训练 + CMSA
- 利用大量无标注 3DGS + 有标注点云数据（94,708 对）进行预训练
- 少量 3DGS 标注（6,631）做微调
- 解决了 3DGS 标注成本高的问题

### 局限性分析

#### 局限1：轻量语言模型限制
- RoBERTa 的词汇覆盖不足导致部分回答错误
- 对复杂指令理解能力有限
- 解决方案：未来可集成更强 LLM（如 LLaMA）提升语言理解和文本生成

#### 局限2：多不连续区域处理
- 对具有多个不连续 affordance 区域的物体（如多层储物柜）预测质量下降
- 可能需要实例级分割或层次化 affordance 建模

#### 局限3：数据集规模和覆盖
- 21 类别仍有限，某些类别样本少（如 Earphone 仅 14 个测试样本）
- 仅在合成/扫描物体上验证，真实世界 3DGS 重建质量影响待评估

### 适用性与场景分析

#### 适用场景
- **静态物体 affordance 分析**：理解单个物体的可操作区域
- **语言引导的 affordance 推理**：根据自然语言指令定位操作区域
- **高保真视觉反馈**：利用 3DGS 的纹理和几何细节增强交互可视化

#### 不适用场景
- **动态/变形物体**：3DGS 重建和 affordance 标注主要针对静态形状
- **实时交互式操作**：当前聚焦识别而非执行，需结合机器人策略
- **开放场景多物体推理**：单物体 affordance 为主

## 与相关论文对比

### LASO - Language-driven Affordance Segmentation

#### 方法对比
| 对比维度 | LASO | 3DAffordSplat |
|----------|------|---------------|
| 核心表示 | 点云 + 语言 | 3DGS + 点云 + 语言 |
| 数据规模 | 8.4k 点云 | 23k GS + 8.4k PC + 6.6k GS标注 |
| 模型输入 | 点云 (xyz) | 3DGS 结构参数 (xyz+rotation+scale) |
| 跨模态 | 语言-点云 | 语言-点云-3DGS |
| 标注质量 | 有噪声（来自 3DAffordanceNet） | 人工精细标注 |

#### 关系分析
- **关系类型**：扩展（extends）— 3DAffordSplat 的点云和文本数据直接基于 LASO
- **本文优势**：更丰富的 3DGS 表示、更高质量标注、跨模态结构对齐

### PointRefer - Language-guided Point Cloud Affordance

#### 关系分析
- **关系类型**：改进（improves）— AffordSplatNet 将 PointRefer 的语言-点云范式扩展到 3DGS 域
- 作为基线在 3DGS 上测试，PointRefer 在 3DAffordSplat 微调后达 mIoU 49.40（Seen）
- AffordSplatNet 的专门架构进一步提升了 13+ mIoU 点

### IAGNet - Image-guided Point Cloud Affordance

#### 关系分析
- **关系类型**：对比（compares）
- IAGNet 在 3DGS 上表现差（Seen mIoU=14.63），因为其依赖图像-点云配对，架构不适应 3DGS 的密集高维输入

## 技术路线定位

### 所属技术路线
本文属于 **3D Affordance Reasoning** 技术路线，核心贡献是"从点云到 3DGS 的模态升级"。

### 技术路线发展历程
```
2D Affordance → 3D PC Affordance → Language-PC Affordance → 3DGS Affordance（本文）
  (CNN)         (3DAffordanceNet)    (LASO/SeqAfford)         (3DAffordSplat)
  2018-2021        2021-2023            2024-2025                 2025
```

### 本文在技术路线中的位置
- **承上**：继承 LASO 的语言-点云范式，利用 ShapeSplat 的 3DGS 资源
- **启下**：为 3DGS affordance 研究提供了数据基准和模型起点，可与 LLM planner 和机器人操作策略集成
- **关键节点**：模态升级的关键转折点——证明了 3DGS 在 affordance 领域的价值

## 未来工作建议

### 作者建议的未来工作
1. 将 affordance reasoning 集成到具身机器人中进行物理交互
2. 更强的语言模型替代 RoBERTa 以提升理解和文本生成

### 基于分析的改进方向
1. **更强的 LLM 集成**：使用 LLaMA/CLIP 等替换 RoBERTa，提升指令理解和泛化
2. **动态场景扩展**：从静态物体扩展到 4D（时序 3DGS）affordance，处理铰接物体运动
3. **多物体场景**：从单物体 affordance 扩展到场景级多物体交互推理
4. **与 VLA 集成**：将 affordance 识别作为 VLA 模型的感知前端，指导动作生成（如 π₀ 系列）
5. **弱监督/零样本**：减少对人工标注的依赖，利用 VLM 的常识推理能力自动发现 affordance

## 我的综合评价

### 价值评分

#### 总体评分
**8.5/10** — 开创性数据集贡献，方法设计扎实，CMSA 机制的发现和分析有深度

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | 首次将 3DGS 引入 affordance reasoning，填补数据集和模型双重空白 |
| 技术质量 | 8/10 | 方法设计合理（粒度自适应+CMSA），但语言模型选择偏弱 |
| 实验充分性 | 8.5/10 | 全面的跨模态迁移实验、消融实验、定性分析，包含有趣的反直觉发现 |
| 写作质量 | 8.5/10 | 结构清晰，动机充分，Task-Specific Overalignment 的分析有洞察 |
| 实用性 | 8/10 | 数据和代码开源，多个下游应用场景有潜力，但距离实际部署还有距离 |

### 重点关注
- **CMSA 的 Task-Specific Overalignment 现象**：预训练对齐对 Seen 场景可能适得其反，这一发现在跨模态迁移学习中具有普遍参考价值
- **仅用结构参数（xyz+rotation+scale）的输入设计选择**：说明 affordance 主要由几何结构决定，外观信息相对次要
- **3DGS vs 点云的模态特性对比**：实验系统性地展示了 3DGS 在 affordance 表示上的优势

## 我的笔记

%%
这篇论文最有趣的地方是 CMSA 的"双面性"发现——预训练对齐对 Unseen 泛化有帮助但对 Seen 场景反而有害。这个 Task-Specific Overalignment 现象在其他跨模态迁移任务中也可能存在。

从具身 AI 的角度看，3DAffordSplat 可以作为 VLA 模型的感知前端：先识别"可操作区域"，再由 VLA 生成具体动作。这是 π₀/Hi Robot 等论文中缺失的能力——它们直接端到端预测动作，跳过了显式的 affordance 推理环节。

数据集构建的"整合"思路（ShapeSplat + LASO）很务实，但受限于两个源数据集的质量和覆盖范围。未来如果能直接从互联网视频中学习 3DGS affordance，才能实现大规模扩展。
%%

## 相关论文

### 直接相关
- LASO - 语言驱动的点云 affordance 分割（数据基础）
- ShapeSplat - 大规模 3DGS 形状数据集（3DGS 数据源）
- 3DAffordanceNet - 首个 3D 点云 affordance 基准

### 背景相关
- PIAD/PIADv2 - 图像-点云 affordance 数据集
- SeqAfford - 序列化 affordance 问答
- LangSplat - 语言引导的 3DGS 分割

### 后续/平行工作
- [[pi0_Vision-Language-Action_Flow_Model|π₀]] - 可集成 affordance 推理作为感知前端
- [[Hi_Robot_Hierarchical_VLA|Hi Robot]] - 层级 VLA 可从 affordance 推理受益
- RT-Affordance - affordance 引导的机器人操作

## 外部资源
- [GitHub](https://github.com/HCPLab-SYSU/3DAffordSplat)
- [Hugging Face Dataset](https://huggingface.co/datasets/Weizm/3DAffordSplat)

> [!tip] 关键启示
> 3DGS 的连续密集表示为 affordance reasoning 带来了质的提升——从离散点到连续曲面的模态升级。CMSA 的"双面性"也提醒我们：预训练对齐不是银弹，需要在泛化和任务特化之间平衡。

> [!warning] 注意事项
> - CMSA 对 Seen 场景可能有负面影响（Task-Specific Overalignment），实际部署需权衡
> - 仅使用结构参数（xyz+rotation+scale）作为输入，不包含颜色/外观
> - 语言模型较弱（RoBERTa），复杂指令理解能力有限
> - 集中在单物体 affordance，场景级多物体推理待探索

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读！如果你是做 affordance/具身 AI 交叉方向，这是必读论文。它为从点云到 3DGS 的模态升级提供了数据和模型的完整参考。
