---
date: "2026-05-12"
paper_id: "arXiv:2506.01953"
title: "Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning"
authors: "Hao Chen, et al."
domain: "VLA_Embodied"
tags:
  - 论文笔记
  - VLA
  - Dual-System
  - System1-System2
  - Diffusion-Policy
  - 3D-Point-Cloud
  - Co-Training
  - Asynchronous-Frequency
  - Real-Time-Control
  - Hi-Robot
  - π0-series
quality_score: "8.8/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning

## 核心信息
- **论文ID**：arXiv:2506.01953
- **作者**：Hao Chen 等（CUHK、北京大学、AI2Robotics、BAAI）
- **机构**：CUHK, Peking University, AI2Robotics, BAAI
- **发布时间**：2025-06
- **会议/期刊**：NeurIPS 2025
- **链接**：[arXiv](https://arxiv.org/abs/2506.01953) | [Project](https://fast-in-slow.github.io/)
- **引用**：--

## 摘要翻译

### 英文摘要
Generalized policy and execution efficiency constitute the two critical challenges in robotic manipulation. While recent foundation policies benefit from the common-sense reasoning capabilities of internet-scale pretrained vision-language models (VLMs), they often suffer from low execution frequency. To mitigate this dilemma, dual-system approaches, inspired by Kahneman's theory, have been proposed to leverage a VLM-based System 2 model handling high-level reasoning and a separate System 1 action model ensuring real-time control. However, existing designs maintain both systems as separate models, limiting System 1 from fully leveraging the rich pretrained knowledge from the VLM-based System 2. In this work, we propose Fast-in-Slow (FiS), a unified dual-system vision-language-action (VLA) model that embeds the System 1 execution module within the VLM-based System 2 by partially sharing parameters. This innovative paradigm not only enables high-frequency execution in System 1, but also facilitates coordination between the reasoning and execution components within a single foundation model of System 2. Given their fundamentally distinct roles within FiS-VLA, we design the two systems to incorporate heterogeneous modality inputs alongside asynchronous operating frequencies, enabling both fast and precise manipulation. To enable coordination between the two systems, a dual-aware co-training strategy is proposed that equips System 1 with action generation capabilities while preserving System 2's contextual reasoning representation. For evaluation, FiS-VLA outperforms previous state-of-the-art methods by 8% in simulation and 11% in real-world tasks in terms of average success rate, while achieving a 117.7 Hz control frequency with action chunk set to eight.

### 中文翻译
通用策略和执行效率是机器人操作中的两个关键挑战。最近的 foundation policy 受益于互联网规模预训练 VLM 的常识推理能力，但通常执行频率较低。为解决这一困境，受 Kahneman 双系统理论启发的方法被提出：基于 VLM 的 System 2 处理高层推理，独立的 System 1 动作模型保证实时控制。然而，现有设计将两个系统保持为独立模型，限制了 System 1 充分利用 VLM 的丰富预训练知识。本文提出 Fast-in-Slow（FiS），一个统一的双系统 VLA 模型，通过部分参数共享将 System 1 执行模块嵌入 VLM 的 System 2 中。这一创新范式不仅使 System 1 实现高频执行，还促进了同一基础模型内推理与执行组件之间的协调。由于两个系统在 FiS-VLA 中承担根本不同的角色，我们为它们设计了异构模态输入和异步操作频率。为实现两系统协调，提出双感知联合训练策略。FiS-VLA 在仿真中超越先前 SOTA 方法 8%，在真实任务中超越 11%，且 action chunk=8 时达到 117.7 Hz 控制频率。

### 核心要点提炼
- **研究背景**：VLA 模型虽具备通用性但推理频率低（~10Hz），双系统方法中 System 1 作为独立轻量模型无法充分利用 VLM 预训练知识
- **研究动机**：能否让 System 1 作为 VLM 的一部分，既继承预训练知识又实现高频控制？
- **核心方法**：将 VLM 最后几个 transformer block 重构为 System 1 执行模块，与完整的 System 2 VLM 共享参数，形成"快在慢中"架构
- **主要结果**：RLBench 平均 69% SR（+8% SOTA），推理速度 21.9Hz（chunk=1），真实双机械臂 68%/74% SR
- **研究意义**：首次实现 System 1/System 2 在同一 VLM 内的统一，证明了嵌入而非附加的架构优势

## 研究背景与动机

### 领域现状
VLA 模型通过微调互联网规模预训练的 VLM 到机器人数据集上，实现了通用操作能力（如 RT-2、OpenVLA、π₀）。但这些数十亿参数模型的推理频率低（通常 <15Hz），难以满足实时闭环控制需求。

受 Kahneman 双系统理论启发，近期工作（CogACT、π₀、HiRT、Helix）提出双系统架构：System 2（VLM）负责高层推理，System 1（独立轻量策略头）负责快速动作生成。然而这些设计中 System 1 作为独立模型，缺乏互联网规模预训练知识，仅依赖 System 2 提取的特征表示。

### 现有方法的局限性
1. **System 1 缺乏预训练知识**：独立策略头仅依赖 System 2 的特征输出，无法直接利用 VLM 的丰富常识推理
2. **两系统独立难以协调**：分离的模型设计限制了推理与执行的深度融合
3. **特征传递瓶颈**：System 2 到 System 1 的信息传递受限于特征表示的压缩

### 研究动机
受神经科学研究启发（人脑中双过程认知共享神经基础），提出核心问题：**"如果 VLM 是机器人的'大脑'，能否在其中集成 System 1 和 System 2 过程，实现协调推理与执行？"**

## 研究问题

### 核心研究问题
1. 如何在保持 VLM 推理能力的前提下，将高频执行模块嵌入同一模型？
2. 两个功能不同的系统如何设计输入模态和操作频率？
3. 如何联合优化推理（System 2）和执行（System 1）组件？

## 方法概述

### 核心思想
**"Fast-in-Slow"架构**：不再为 System 1 设计独立模型，而是将 VLM（System 2）的最后几个 transformer block 重构为高频执行模块。这类似于人脑 — 快思考和慢思考共享同一神经基础，而非两个独立大脑。

### 方法框架

#### 整体架构

![[method_fig.pdf|800]]

> 图1：FiS-VLA 整体框架。完整 VLM 作为 System 2 处理 2D 图像和语言指令，最后的 transformer block 被重构为 System 1 执行模块，处理 3D 点云、2D 图像和机器人状态。

**架构核心**：
- System 2 = 完整 VLM（32 层 LLaMA2），低频率运行
- System 1 = VLM 的最后 2 层 transformer block，高频率运行
- 两系统共享视觉编码器和大部分 LLM 参数
- System 2 的中间层潜在特征作为 System 1 的条件输入

#### 各模块详细说明

**视觉编码器**
- **功能**：提取 2D 图像和 3D 点云特征
- **设计**：SigLIP（语义特征） + DINOv2（局部空间细节）双编码器，特征拼接
- **输入**：224×224 图像
- **输出**：$f^{\mathrm{SigLIP}} \in \mathbb{R}^{N_v \times 1024}$ + $f^{\mathrm{DINO}} \in \mathbb{R}^{N_v \times 1152}$

**3D 点云编码器**
- **功能**：高效处理点云输入，提取空间几何特征
- **设计**：轻量 3D tokenizer（FPS 降采样 → KNN 局部聚合 → Linear 特征编码），然后通过共享视觉编码器提取特征
- **优势**：
  1. 利用 VLM 已有的视觉-语言对齐能力投影 3D 信息
  2. 避免大幅参数增加，保持计算效率
- **输入**：单视角深度图反投影的 1024 个点
- **关键设计**：不引入新 3D 编码器，而是复用视觉编码器处理 tokenized 点云

**LLM 骨干网络**
- **模型**：7B LLaMA2，32 层 decoder-only transformer
- **System 2**：完整 32 层，负责多模态理解和推理
- **System 1**：最后 2 层（通过消融实验确定），负责扩散动作生成
- **关键设计**：System 2 的中间层（第 30 层）输出作为潜在条件传递给 System 1

**MLP 辅助组件**
- 视觉-语言投影器：将 2D/3D 特征映射到 LLM 文本嵌入空间
- 状态编码器：编码机器人本体状态
- 时间步/噪声动作投影器：用于扩散过程

### 双系统协调机制

#### 异步频率设计

核心研究问题：**System 2 的中间理解输出能有效指导多少未来动作步？**

- **System 2（慢）**：低频率处理 2D 图像 + 语言指令 → 产生中间层潜在特征 → 作为 System 1 的时序条件
- **System 1（快）**：每个时间步利用最新观测 + System 2 周期性更新的潜在特征 → 实时生成动作
- **最优频率比**：1:4（System 2 : System 1），即 System 2 每 4 步更新一次
- **训练时**：采用异步采样降低 System 2 运行频率，鼓励 System 1 保持时序一致性

当 action chunk=8 时，FiS-VLA 在 NVIDIA 4090 上达到 **117.7 Hz** 控制频率。

#### 异构模态输入

| 系统 | 模态 | 理由 |
|------|------|------|
| System 2 | 2D 图像 + 语言指令 | 利用 VLM 互联网图文预训练知识进行高层语义推理 |
| System 1 | 3D 点云 + 2D 图像 + 机器人状态 | 全面理解当前环境，支持精确空间操作和闭环控制 |

消融实验验证了每种模态对 System 1 都有实质贡献：3D 点云增强几何空间理解，机器人状态提供本体感知，2D 图像保持视觉感知连续性。

### 训练目标与策略

#### 双感知联合训练（Dual-Aware Co-Training）

为避免仅训练扩散动作生成导致 System 2 自回归推理能力的灾难性遗忘，提出联合训练目标：

**System 1 — 扩散去噪损失**：
$$\mathcal{L}_{\text{fast}} = \mathbb{E}_{\tau, c, \tilde{a}, \eta} \left[ \| \eta - \pi_{\theta_f} (\sqrt{\beta_\tau} \tilde{a} + \sqrt{1 - \beta_\tau} \eta, c, \tau) \|^2 \right]$$

条件 $c$ 包含两部分：System 2 的低频潜在特征 + System 1 的高频输入。

**System 2 — 自回归交叉熵损失**：
$$\mathcal{L}_{\text{slow}} = - \sum_{i=1}^{D_t} \log P(\hat{a}_{i} \mid \text{context}, \theta)$$

监督信号可以是离散动作 token 或语言子任务计划。

**总损失**：
$$\mathcal{L}_{\text{FiS-VLA}} = \mathcal{L}_{\text{fast}} + \mathcal{L}_{\text{slow}}$$

#### 训练流程

1. **预训练**：在 860K+ 轨迹（OXE + DROID + ROBOMIND 等）上训练 5 epoch，仅使用单图像观测，System 2 以离散动作序列监督
2. **微调**：在高质量自采数据（仿真 RLBench + 真实双机械臂）上微调，引入语言子任务标注增强 System 2 目标

## 实验结果

### 实验目标
全面评估 FiS-VLA 的操作成功率、推理速度和泛化能力。

### 数据集

#### 仿真基准：RLBench（10 个任务）

| 任务 | 类型 |
|------|------|
| Close box | 关 |
| Close Laptop | 关 |
| Toilet seat down | 关 |
| Sweep to dustpan | 扫 |
| Close fridge | 关 |
| Phone on base | 放 |
| Take umbrella out | 取 |
| Frame off hanger | 取 |
| Wine at rack | 放 |
| Water plants | 浇 |

每任务 100 条轨迹，Franka Panda 机械臂，前端相机，关键帧采样。

#### 真实世界：双机械臂（8 个任务）

| 平台 | 控制模式 | 任务 |
|------|----------|------|
| AgileX | 末端位姿控制（14-DoF） | Pick & place, Lift ball & place, Place bottles at rack, Wipe blackboard |
| AlphaBot | 关节位置控制（16-DoF） | Pick bowl & place, Handover & place, Pour water & move, Fold towel & place |

每任务 100 条遥操作演示，三视角相机。

### 实验设置

#### 基线方法
- **ManipLLM**：VLA 方法
- **OpenVLA**：VLA 方法
- **π₀**：双系统、同步频率，2.6B LLM
- **CogACT**：双系统、同步频率，7B LLM

#### 实验环境
- 训练：8× NVIDIA A800 GPU，AdamW 优化器，300 epoch，混合精度
- 推理：NVIDIA 4090 GPU

### 主要结果

#### 仿真结果（RLBench）

| 模型 | Close box | Close laptop | Toilet seat | Sweep | Close fridge | Phone | Umbrella | Frame | Wine | Water | **平均 SR** | 推理速度 |
|------|-----------|-------------|-------------|-------|-------------|-------|----------|-------|------|-------|-------------|----------|
| ManipLLM | 0.50 | 0.80 | 0.40 | 0.20 | 0.80 | 0.35 | 0.10 | 0.25 | 0.15 | 0.20 | 0.38 | 2.2 Hz |
| OpenVLA | 0.65 | 0.40 | 0.75 | 0.50 | 0.80 | 0.20 | 0.35 | 0.15 | 0.10 | 0.10 | 0.40 | 6.3 Hz |
| π₀ | 0.90 | 0.80 | **0.95** | 0.30 | 0.85 | 0.30 | 0.30 | **0.70** | 0.10 | **0.30** | 0.55 | 13.8 Hz |
| CogACT | 0.90 | 0.80 | **0.95** | 0.50 | 0.85 | **0.50** | **0.55** | 0.45 | 0.30 | 0.25 | 0.61 | 9.8 Hz |
| **FiS-VLA** | **1.00** | **1.00** | **0.95** | **0.55** | **0.90** | **0.50** | 0.50 | **0.70** | **0.55** | 0.20 | **0.69** | **21.9 Hz** |

**关键发现**：
- 平均 69% SR，超越 CogACT 8%、π₀ 14%
- 10 个任务中 8 个最优
- **推理速度 21.9 Hz**，是 CogACT（9.8 Hz）的 2.2 倍、π₀（13.8 Hz）的 1.6 倍
- 注意 π₀ 使用 2.6B LLM，而 FiS-VLA 使用 7B LLM

#### 真实世界结果

![[realworld_fig.pdf|800]]

| 模型 | AgileX 平均 SR | AlphaBot 平均 SR |
|------|---------------|------------------|
| π₀ | 0.59 | 0.61 |
| **FiS-VLA** | **0.68** | **0.74** |

真实场景平均超越 11%，复杂操作任务（如 Bottles at Rack、Fold Towel）优势明显。

### 消融实验

![[ablation_fig.pdf|800]]

> 图2：消融实验。(1) System 1 共享层数，(2) System 1 输入模态组合，(3) 双系统操作频率比。

**（1）System 1 共享层数**：1→2 层性能明显提升，2 层后趋于饱和。用 2 层即可实现稳定操作和高推理速度。

**（2）输入模态贡献**：仅潜在特征 < +机器人状态 < +2D 图像 < +3D 点云。各模态均有实质贡献，3D 点云对空间推理任务提升最显著。

**（3）频率比**：1:4 最优，在慢推理和快执行间取得最佳平衡。过低（1:1）失去异步优势，过高（1:8）System 2 更新太稀疏。

**（4）训练策略**：移除 $\mathcal{L}_{\text{slow}}$ 后 SR 从 69% 降至 62%，验证了双感知联合训练的必要性。

### 泛化实验

![[generalization_fig.pdf|400]]

| 测试场景 | FiS-VLA (AgileX) | π₀ (AgileX) | FiS-VLA (AlphaBot) | π₀ (AlphaBot) |
|----------|-----------------|-------------|--------------------|---------------|
| 原始 | 0.70 | 0.55 | 0.80 | 0.65 |
| 未见物体 | 0.55 (-21%) | 0.40 (-27%) | 0.65 (-19%) | 0.40 (-38%) |
| 复杂背景 | 0.50 (-29%) | 0.35 (-36%) | 0.60 (-25%) | 0.40 (-38%) |
| 光照变化 | 0.50 (-29%) | 0.40 (-27%) | 0.55 (-31%) | 0.35 (-46%) |

FiS-VLA 在所有泛化场景中性能下降幅度均小于 π₀，特别是在 AlphaBot 未见物体测试中（-19% vs -38%）。

## 深度分析

### 研究价值评估

#### 理论贡献
- **统一双系统 VLA 架构**：首次将 System 1 嵌入而非附加到 VLM 中，实现参数共享和深度融合
  - 创新点：从"附加独立策略头"到"复用 VLM 已有层"的范式转变
  - 学术价值：为 VLA 架构设计提供了新方向
- **异构模态 × 异步频率的系统性设计**：为两个功能不同的系统匹配合适的输入模态和运行频率
- **双感知联合训练策略**：通过扩散去噪 + 自回归联合优化，防止灾难性遗忘

#### 实际应用价值
- 在 7B 参数模型上实现 21.9Hz（chunk=1）和 117.7Hz（chunk=8）的推理速度
- 支持两种机器人平台（AgileX、AlphaBot）和两种控制模式（位姿、关节位置）
- 对未见物体、复杂背景、光照变化均有较强泛化能力

### 方法优势详解

#### 优势1：嵌入 vs 附加
- **技术基础**：System 1 复用 VLM 最后几层而非独立模块
- **实验验证**：2 层共享即可达到最优性能，参数效率高
- **对比**：相比 π₀ 的独立 action expert、CogACT 的独立策略头，FiS 的嵌入设计使 System 1 直接继承 VLM 预训练知识

#### 优势2：异步频率设计
- **技术基础**：System 2 低频更新潜在条件，System 1 每步执行
- **实验验证**：1:4 频率比下既提升速度（21.9Hz）又提升精度（69% SR）
- **关键洞察**：降低 System 2 频率不仅提速，还增加了 System 1 接收的观测信息丰富度

#### 优势3：异构模态输入
- 2D 给 System 2（语义），3D+2D+State 给 System 1（精确控制）
- 验证了"不同功能的系统需要不同模态"的设计原则

### 局限性分析

#### 局限1：单 GPU 推理限制
- **描述**：当前无法像 Helix 那样将两个系统部署在不同 GPU 上并行推理
- **影响**：频率比的实际收益受限于串行执行
- **解决方案**：未来可探索多 GPU 部署

#### 局限2：System 2 更新稀疏
- **描述**：1:4 频率比意味着 System 2 每 4 步才更新一次对场景的理解
- **表现**：可能需要更智能的 System 2 触发机制（而非固定频率）

#### 局限3：仅探索了 Simulation + 有限真实任务
- **描述**：真实实验为单任务设置，多任务真实场景未验证
- **数据集规模**：虽然预训练数据量大（860K），但微调数据相对有限

### 适用性与场景分析

#### 适用场景
- **需要高频控制的操作任务**：动态物体操作、力控装配
- **多模态输入场景**：有深度相机/3D 传感器的机器人平台
- **需要语义理解和精确执行的复杂任务**：如"把瓶子放到架子上"

#### 不适用场景
- **纯导航任务**：无需精细操作，VLA 架构可能过重
- **极低算力平台**：7B 模型需要 GPU

## 与相关论文对比

### [[Hi_Robot_Hierarchical_VLA|Hi Robot]] - Hierarchical VLA

#### 方法对比
| 对比维度 | Hi Robot | FiS-VLA |
|----------|----------|---------|
| 核心思想 | 两个独立 VLM 分别做 System 1/2 | 同一 VLM 内嵌 System 1 |
| System 1 设计 | 独立的动作 VLM（小模型） | VLM 最后几层（共享参数） |
| 模型分离度 | 完全分离 | 部分共享 |
| 训练方式 | 两阶段训练 | 联合训练（双感知） |
| 层级关系 | 垂直层级（高层→低层指令） | 水平嵌入（同一模型内快慢并行） |

#### 关系分析
- **关系类型**：平行探索（related）
- **互补性**：Hi Robot 更侧重高层指令分解和交互，FiS-VLA 更侧重底层执行的架构创新

### [[pi0_Vision-Language-Action_Flow_Model|π₀]] - Flow Model VLA

#### 方法对比
| 对比维度 | π₀ | FiS-VLA |
|----------|-----|---------|
| System 1 设计 | 独立 Action Expert（transformer） | 共享 VLM 最后几层 |
| 动作生成 | Flow Matching | Diffusion（DDPM） |
| 频率设计 | 同步 | 异步（1:4） |
| 参数共享 | 无（Action Expert 独立） | 部分共享 |
| 推理速度 | 13.8 Hz | 21.9 Hz |

#### 关系分析
- **关系类型**：改进（improves）— FiS-VLA 可视为对双系统 VLA 架构的改进
- **本文优势**：推理速度更快、参数利用更高效、System 1 继承预训练知识

### [[KI_Knowledge_Insulation_for_VLA|KI]] - Knowledge Insulation

#### 关系分析
- **关系类型**：概念相关（related）
- **共同点**：都关注 VLM 预训练知识的保留和利用
- **差异**：KI 关注联合训练中的知识隔离（stop-gradient），FiS-VLA 关注架构层面的知识共享

### [[RTC_Real_Time_Action_Chunking|RTC]] - Real-Time Chunking

#### 关系分析
- **关系类型**：补充（related）
- **共同点**：都关注 VLA 的实时推理问题
- **差异**：RTC 是推理时算法（inpainting），FiS-VLA 是架构级解决方案；两者可互补

## 技术路线定位

### 所属技术路线
本文属于 **双系统 VLA** 技术路线，核心特点：
- 从"附加独立 System 1"到"嵌入 System 1 于 VLM 中"
- 异步频率设计实现高频控制
- 异构模态匹配不同系统功能

### 技术路线发展历程
```
VLA 诞生（RT-2） → 双系统分离（π₀/CogACT/HiRT） → 双系统嵌入（FiS-VLA） → 端到端统一？
        ↑                    ↑                              ↑
    2023                 2024-2025                       2025
```

### 本文在技术路线中的位置
- **承上**：继承 π₀ 的双系统思想、OpenVLA 的 VLM 微调范式、扩散策略的动作生成
- **启下**：为"统一 VLA 模型内的多系统协调"提供了第一个可行方案
- **关键节点**：范式转变 — 从"加一个模块"到"复用已有模块"

## 未来工作建议

### 作者建议的未来工作
1. **多 GPU 并行部署**：将两个系统部署到不同 GPU 进一步提升速度
2. **扩展到更多 embodiment**：验证在不同机器人形态上的通用性

### 基于分析的未来方向
1. **自适应频率调节**：根据任务复杂度动态调整 System 2 更新频率，而非固定 1:4
2. **与 RTC 结合**：将 FiS-VLA 架构与推理时 inpainting 算法结合，进一步提升实时性
3. **更多模态探索**：触觉、力矩等接触式传感器作为 System 1 输入
4. **多任务真实世界验证**：扩展到真实多任务设置

## 我的综合评价

### 价值评分

#### 总体评分
**8.8/10** — 架构创新显著，实验充分，兼具理论和实用价值

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | "嵌入而非附加"的范式转变，首次实现统一 VLM 内的双系统协调 |
| 技术质量 | 8.5/10 | 方法设计系统性强（架构→模态→频率→训练），消融实验全面 |
| 实验充分性 | 8.5/10 | 仿真 10 任务 + 真实 8 任务 + 泛化 3 场景，基线选择合理 |
| 写作质量 | 8.5/10 | 清晰的 motivation，图/表质量高，技术细节完整 |
| 实用性 | 9/10 | 21.9Hz 推理，支持两种控制模式，可直接部署 |

### 重点关注
- **嵌入 vs 附加的架构设计哲学**：这可能是影响未来 VLA 架构设计的关键洞察
- **消融实验的完备性**：对层数、模态、频率比、训练策略四个维度的系统消融
- **真实世界泛化实验**：三种泛化场景的评估展示了方法的鲁棒性

## 我的笔记

%%
FiS-VLA 最打动我的是架构设计的"减法"思路：不是加更多模块（如独立 System 1），而是发掘 VLM 内部已有的能力。将最后几层重构为执行模块，既减少参数冗余，又让执行继承了推理的预训练知识。

与 Hi Robot 对比很有意思：Hi Robot 是"两个大脑"思路（两个独立 VLM），FiS-VLA 是"一个大脑两个模式"思路。前者分工清晰但信息传递有损失，后者信息传递高效但模式切换机制还需探索。

一个未讨论的问题：如果 System 2 的潜在特征不够好（因为预训练知识不足），System 1 是否也会受影响？KI 的知识隔离方法是针对联合训练的，但 FiS-VLA 的架构级共享可能让这个问题更难处理。
%%

## 相关论文

### 直接相关
- [[Hi_Robot_Hierarchical_VLA|Hi Robot]] - 双系统 VLA，层级指令跟随
- [[pi0_Vision-Language-Action_Flow_Model|π₀]] - 双系统架构，flow matching 动作生成
- [[KI_Knowledge_Insulation_for_VLA|KI]] - VLA 联合训练中知识隔离

### 背景相关
- [[RTC_Real_Time_Action_Chunking|RTC]] - 实时动作分块推理
- CogACT - 双系统认知动作生成
- OpenVLA - 开源 VLA 基础模型

### 后续工作
- 多 GPU 并行部署的 FiS-VLA
- 自适应频率调节的 FiS-VLA 变体

## 外部资源
- [Project Page](https://fast-in-slow.github.io/) — 包含视频和代码

> [!tip] 关键启示
> System 1 不需要是一个独立模型 — 复用 VLM 的最后几层比附加新模块更高效，这启示我们：在 VLA 设计中，先问"VLM 内部能否解决"再问"需要加什么"。

> [!warning] 注意事项
> - 最优频率比 1:4 是在 RLBench 任务上得出的，不同任务可能需要不同比例
> - 当前仅在 7B LLaMA2 上验证，其他 VLM 骨干的适用性待验证
> - System 1 需要 3D 点云输入，对传感器配置有要求

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！这是双系统 VLA 架构设计的重要创新，对理解 VLA 中推理与执行的关系有启发性。
