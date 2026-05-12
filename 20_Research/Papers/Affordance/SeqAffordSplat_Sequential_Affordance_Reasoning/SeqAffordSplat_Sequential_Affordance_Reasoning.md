---
date: "2026-05-12"
paper_id: "arXiv:2507.23772"
title: "SeqAffordSplat: Scene-level Sequential Affordance Reasoning on 3D Gaussian Splatting"
authors: "Di Li, Jie Feng, Jiahao Chen, Weisheng Dong, Guanbin Li, Yuhui Zheng, Mingtao Feng, Guangming Shi"
domain: "Affordance"
tags:
  - 论文笔记
  - Affordance
  - 3DGS
  - 3D-Gaussian-Splatting
  - Sequential-Reasoning
  - Scene-Level
  - LLM
  - VFM
  - Multi-Modal
  - Embodied-AI
  - Long-Horizon
  - AAAI-2026
quality_score: "8.8/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# SeqAffordSplat: Scene-level Sequential Affordance Reasoning on 3D Gaussian Splatting

## 核心信息
- **论文ID**：arXiv:2507.23772
- **作者**：Di Li, Jie Feng, Jiahao Chen, Weisheng Dong, Guanbin Li, Yuhui Zheng, Mingtao Feng, Guangming Shi（西安电子科技大学、中山大学、青海师范大学）
- **机构**：Xidian University, Sun Yat-sen University, Qinghai Normal University
- **发布时间**：2025-07-31
- **会议/期刊**：AAAI 2026（投稿）
- **链接**：[arXiv](https://arxiv.org/abs/2507.23772)
- **引用**：--

## 摘要翻译

### 英文摘要
3D affordance reasoning, the task of associating human instructions with the functional regions of 3D objects, is a critical capability for embodied agents. Current methods based on 3D Gaussian Splatting (3DGS) are fundamentally limited to single-object, single-step interactions, a paradigm that falls short of addressing the long-horizon, multi-object tasks required for complex real-world applications. To bridge this gap, we introduce the novel task of Sequential 3D Gaussian Affordance Reasoning and establish SeqAffordSplat, a large-scale benchmark featuring 1800+ scenes to support research on long-horizon affordance understanding in complex 3DGS environments. We then propose SeqSplatNet, an end-to-end framework that directly maps an instruction to a sequence of 3D affordance masks. SeqSplatNet employs a large language model that autoregressively generates text interleaved with special segmentation tokens, guiding a conditional decoder to produce the corresponding 3D mask. To handle complex scene geometry, we introduce a pre-training strategy, Conditional Geometric Reconstruction, where the model learns to reconstruct complete affordance region masks from known geometric observations, thereby building a robust geometric prior. Furthermore, to resolve semantic ambiguities, we design a feature injection mechanism that lifts rich semantic features from 2D Vision Foundation Models (VFM) and fuses them into the 3D decoder at multiple scales. Extensive experiments demonstrate that our method sets a new state-of-the-art on our challenging benchmark, effectively advancing affordance reasoning from single-step interactions to complex, sequential tasks at the scene level.

### 中文翻译
3D affordance reasoning 是将人类指令与 3D 物体功能区域关联的关键具身 AI 能力。现有基于 3DGS 的方法局限于单物体、单步交互，无法应对真实场景中的长序列、多物体任务。为填补这一空白，我们提出新任务——Sequential 3D Gaussian Affordance Reasoning，并建立 SeqAffordSplat 基准（1800+ 场景，14000+ affordance mask，8000+ 序列指令）。我们提出 SeqSplatNet，端到端框架将指令直接映射为 3D affordance mask 序列。SeqSplatNet 使用 LLM 自回归生成文本与特殊分割 token 交错序列，引导条件解码器生成对应 3D mask。为处理复杂场景几何，提出条件几何重建预训练策略（从几何观测中学习重建完整 affordance mask）。为解析语义歧义，设计 VFM 语义特征注入机制（从 2D VFM 提取语义特征融合到 3D 解码器多尺度层）。大量实验表明我们的方法在挑战性基准上达到 SOTA，将 affordance reasoning 从单步交互推进到场景级序列任务。

### 核心要点提炼
- **研究背景**：3DAffordSplat 等现有方法仅支持单物体、单步 affordance reasoning，无法处理"操作微波炉加热碗中食物"这类需要多步序列的场景级任务
- **研究动机**：真实世界指令的高级简洁性要求 affordance 系统具备：(i) 组合原始指令序列的能力，(ii) 在复杂场景中跨物体动态切换可操作区域的能力
- **核心方法**：LLM 自回归生成 ⟨SEG⟩ token 驱动的序列 mask 预测 + 条件几何重建预训练 + VFM 多视图语义特征提升
- **主要结果**：单步 mIoU=37.0（+6.5），端到端序列 sIoU=26.2（+14.1，是 SeqAfford 的 2.2 倍），3DAffordSplat 数据集上 mIoU=40.2（+9.9）
- **研究意义**：首次将 affordance reasoning 从单步/单物体推进到场景级序列推理，打通了从指令理解到多步操作规划的端到端路径

## 研究背景与动机

### 领域现状
3D affordance reasoning 经历了从点云到 3DGS 的模态升级路径：
- **点云时代**：3DAffordanceNet → LASO → SeqAfford（首个序列 affordance，但仍在点云上，定位精度不足）
- **3DGS 时代**：3DAffordSplat（首个 3DGS affordance 数据集+模型，但仅限于单物体单步交互）

当前的核心 gap：**现有 3DGS affordance 方法只能处理"什么可以操作"（what），无法处理"按什么顺序操作"（in what order）**。

### 现有方法的局限性
- 3DAffordSplat/PointRefer/IAGNet：每个场景仅含单个物体，每条指令只对应一个原子动作
- SeqAfford：支持序列推理但基于稀疏点云，定位精度不足
- 缺少同时具备 (1) 3DGS 高保真表示、(2) 场景级复杂环境、(3) 序列化推理 的框架

### 研究动机
真实世界指令如"用微波炉加热碗中食物"涉及多个跨物体的连续操作步骤。这要求 affordance 系统既能理解长序列指令的因果依赖，又能在复杂场景中精确定位每个步骤的可操作区域。现有方法的 constrained task prototype 无法支撑此类应用。

## 研究问题

### 核心研究问题
1. 如何定义和构建场景级序列 3DGS affordance reasoning 任务与基准？
2. 如何在统一架构中实现指令理解、序列规划和 3D 精确定位？
3. 如何在复杂场景几何中为 affordance reasoning 提供鲁棒的几何和语义先验？

## 方法概述

### 核心思想
**LLM 作为序列推理引擎 + 3DGS 作为高保真表示**。SeqSplatNet 的核心创新在于：LLM 自回归生成过程中，每输出一个特殊 ⟨SEG⟩ token 就触发一次 affordance mask 解码，实现"边推理边定位"的端到端统一框架。

### 方法框架

#### 整体架构

![[pipeline_v2.pdf|800]]

> 图1：SeqSplatNet 架构。包含三个核心组件：(1) 3DGS Encoder 提取几何特征，(2) LLM 自回归生成指令序列+⟨SEG⟩ token，(3) Conditional Affordance Decoder 根据每个 ⟨SEG⟩ 的隐藏状态解码对应 3D affordance mask。两个增强模块：条件几何重建预训练 + VFM 语义特征注入。

#### 任务定义

给定 3DGS 场景 $\mathcal{G}$ 和简洁指令 $Q_{inst}$，预测有序的 $T$ 个二值 affordance mask 序列 $\mathcal{M}=(M_{1}, M_{2},...,M_{T})$。

关键扩展：从"识别有哪些 affordance"到"推理应以什么顺序实现"。

#### 各模块详细说明

**3DGS Encoder**
- 基于 PointNet++，处理 Gaussian 几何属性（position + rotation + scale）
- 输出逐点几何特征 $F_{\text{geo}} \in \mathbb{R}^{N \times d}$（$d=512$）
- 与 3DAffordSplat 一致，仅使用结构参数

**LLM（序列推理引擎）**
- 选用 Qwen3-0.6B，通过 LoRA 微调（rank=8, alpha=16，target: q_proj/k_proj/v_proj/lm_head）
- 增强词汇表：添加特殊 token $\langle \text{SEG} \rangle$
- 自回归生成交错序列：语言 token → $\langle \text{SEG} \rangle$ → 语言 token → $\langle \text{SEG} \rangle$ → ...
- 每个 $\langle \text{SEG} \rangle$ 的隐藏状态 $h_{\text{seg}} \in \mathbb{R}^d$ 编码了来自指令和前置原语的上下文依赖
- 关键优势：masked attention 确保每个 $h_{\text{seg}}$ 只依赖其之前的上下文（因果性），自然捕获序列依赖

**Conditional Affordance Decoder**
- 基于 query-based segmentation 范式
- 每个 LLM 派生的 $h_{\text{seg}}$ 作为潜在查询，从编码的几何特征和注入的语义信息中解码对应的 3D affordance mask $M_t$
- 端到端可微，统一推理与感知

**训练目标**：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{lang}} + \lambda_{\text{mask}} \sum_{t=1}^T \mathcal{L}_{\text{mask}}$$

其中 $\mathcal{L}_{\text{mask}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$（仅对 $\langle \text{SEG} \rangle$ token 位置激活）。

### 条件几何重建预训练（Conditional Geometric Reconstruction）

**动机**：从头训练有效的场景级 3DGS encoder 需要大量标注样本。

**核心思想**：通过从抽象语义嵌入重建空间 affordance 区域，为 3DGS encoder 注入几何先验。

**方法**：
1. 双编码器设计：3DGS encoder $\Phi_{\text{enc}}$（场景→几何特征）+ Mask encoder $\Phi_{\text{mask}}$（GT mask→条件嵌入 $e_{\text{mask}}$）
2. $e_{\text{mask}}$ 作为 query 对几何特征做交叉注意力 → 融合特征 $F_{\text{fused}}$
3. Decoder $\Phi_{\text{dec}}$ 重建完整 mask
4. 关键设计：通过以 mask 嵌入为条件，模型学习将 affordance 语义概念与几何结构解耦表示

**效果**：预训练后 sIoU 从 20.3 提升至 24.1（+3.8）。

### VFM 语义特征注入

**动机**：纯几何表示无法解析复杂场景中的语义歧义（如"可抓握的把手"vs"可推的门面"）。

**方法**：
1. 多视图渲染：从 $m$ 个视角渲染 RGB 图像 → 冻结的 VFM（DINOv2）提取 2D 特征图
2. 无学习特征提升（Learning-free Lifting）：通过 alpha-blending 权重加权平均，将 2D 特征反投影到 3D Gaussians：
   $$f_i^{\text{sem}} = \frac{\sum_{(v, p) \in \mathcal{S}_i} w_i(v, p) F_{p}^{(v)}}{\sum_{(v, p) \in \mathcal{S}_i} w_i(v, p)}$$
3. 多尺度注入：提升后的 $F_{\text{sem}}$ 在 Conditional Affordance Decoder 的多个尺度上以加法融合

**效果**：DINOv2 注入后 sIoU 从 24.1 提升至 26.2（+2.1）；CLIP 注入仅 +0.1（24.2），说明局部空间语义（DINOv2）比全局语义（CLIP）对 affordance 定位更有价值。

## 实验结果

### SeqAffordSplat 数据集

![[teaser_v6.pdf|800]]

> 图2：Teaser — 左侧为任务示意，中间为数据集统计，右侧为模型性能对比。

| 属性 | 数值 |
|------|------|
| 3DGS 场景 | 1800+ |
| Affordance mask | 14,000+ |
| 序列指令 | 8,000+ |
| 物体类别 | 21 |
| Affordance 类型 | 18 |

**数据集构建**：
1. **3DGS 场景生成**：从 3DAffordSplat 取单物体 Gaussian，通过平移/旋转/缩放组合成多物体场景
2. **指令标注**：GPT-4o 生成（提供视觉上下文+文本上下文+角色提示+目标规范）→ 人工验证 → 半自动 affordance 标注迁移

**三种评估设置**：
- **Single**：单步指令 → 单个 mask
- **Sequential（w/ GT seq）**：给定 GT 子指令序列 → 预测 mask 序列（隔离感知模块）
- **Sequential**：仅给定高层指令 → 端到端推理+预测完整序列

**序列评估指标**：sIoU/sAUC/sSIM/sMAE — 先对齐序列长度再计算标准指标（短序列用空帧填充），固有惩罚序列长度不匹配。

### 主要结果

#### SeqAffordSplat 基准结果

| 设置 | 方法 | 来源 | mIoU/sIoU↑ | AUC/sAUC↑ | SIM/sSIM↑ | MAE/sMAE↓ |
|------|------|------|-----------|-----------|-----------|-----------|
| Single | 3DAffordSplat | ACM MM'25 | 30.5 | 92.7 | 0.395 | 0.065 |
| Single | PointRefer | CVPR'24 | 31.3 | 92.1 | 0.411 | 0.055 |
| Single | **SeqSplatNet** | -- | **37.0** | **94.0** | **0.470** | **0.049** |
| Seq(w/ GT) | PointRefer | CVPR'24 | 30.3 | 91.2 | 0.418 | 0.055 |
| Seq(w/ GT) | **SeqSplatNet** | -- | **36.0** | **95.6** | **0.457** | **0.036** |
| Sequential | SeqAfford | CVPR'25 | 12.1 | 73.0 | 0.122 | 0.230 |
| Sequential | **SeqSplatNet** | -- | **26.2** | **80.6** | **0.312** | **0.132** |

**关键发现**：
- 单步 SOTA：mIoU=37.0，超越 PointRefer +5.7、3DAffordSplat +6.5
- 端到端序列：sIoU=26.2，是 SeqAfford 的 **2.2 倍**（+14.1）
- 即使在给定 GT 序列的设置下，SeqSplatNet 的感知模块仍大幅领先

#### 3DAffordSplat 数据集交叉验证

| 方法 | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|------|-------|------|------|------|
| 3DAffordSplat | 30.3 | 83.9 | 0.440 | 0.210 |
| PointRefer | 18.4 | 78.5 | 0.430 | 0.200 |
| **SeqSplatNet** | **40.2** | **89.3** | **0.530** | **0.169** |

在已有基准上同样大幅领先（+9.9 mIoU），证明方法具有跨基准泛化性。

### 消融实验

#### 核心组件消融

| 预训练 | 特征注入 | sIoU↑ | sAUC↑ | sSIM↑ | sMAE↓ |
|--------|----------|-------|-------|-------|-------|
| ✗ | ✗ | 20.3 | 76.3 | 0.229 | 0.169 |
| ✓ | ✗ | 24.1 | 78.5 | 0.302 | 0.141 |
| ✓ | CLIP | 24.2 | 79.1 | 0.290 | 0.141 |
| ✓ | **DINOv2** | **26.2** | **80.6** | **0.312** | **0.132** |

- 条件几何重建预训练贡献最大：**+3.8 sIoU**
- DINOv2 注入额外贡献 +2.1 sIoU；CLIP 仅 +0.1——说明 affordance 定位更需要局部空间语义（DINOv2 擅长）而非全局语义

#### LLM 骨干消融

| LLM | 参数量 | sIoU↑ | sAUC↑ | sSIM↑ | sMAE↓ |
|-----|--------|-------|-------|-------|-------|
| GPT2-small | 0.1B | 12.1 | 43.9 | 0.156 | 0.488 |
| **Qwen3-0.6B** | 0.6B | **26.2** | **80.6** | **0.312** | **0.132** |
| Qwen3-1.7B | 1.7B | 26.4 | 79.5 | 0.291 | 0.147 |
| Qwen3-8B | 8B | 24.2 | 78.6 | 0.285 | 0.148 |

**反直觉发现**：更大模型 ≠ 更好性能。Qwen3-8B 性能反而下降（24.2 vs 26.2），可能原因：大模型对 LoRA 微调响应不足、过参数化导致训练不充分、或任务特定知识与通用语言能力之间存在 trade-off。

### 定性结果

![[visual.pdf|800]]

> 图3：可视化结果。(a) 单步场景：SeqSplatNet 精确定位"bag 的可提起部分"，对比方法错误分割；(b) 序列场景：SeqSplatNet 成功将"用笔记本电脑听音乐"分解为多步 affordance 序列，基线方法无法生成连贯计划。

## 深度分析

### 研究价值评估

#### 理论贡献
- **新任务定义**：首次正式定义"场景级序列 3DGS affordance reasoning"，建立了从单步到序列的范式转变
- **三合一统一框架**：LLM 推理 + 3DGS 感知 + 序列规划在单一端到端架构中的统一
- **⟨SEG⟩ token 机制**：将序列推理自然嵌入 LLM 自回归过程，每个 token 触发一次 mask 解码——这是一个优雅的设计范式，可推广到其他多模态序列任务

#### 实际应用价值
- 直接支撑复杂具身任务：如"清理餐桌"→[拿起杯子→放到水槽→擦拭桌面→摆放餐具]
- 与 VLA 模型天然互补：SeqSplatNet 负责"识别什么可按什么顺序操作"，VLA 负责"如何操作"

### 方法优势详解

#### 优势1：⟨SEG⟩ Token 的优雅设计
- LLM 的自回归因果性自然捕获序列依赖
- 每个 ⟨SEG⟩ 动态生成，支持任意长度序列预测
- 通过 masked attention 确保每步只依赖前置上下文
- 无需显式层次分解或固定序列长度

#### 优势2：条件几何重建预训练
- 自监督学习，无需额外标注
- 通过"给定语义概念→重建空间位置"的任务，学习解耦的语义-几何表示
- 为下游任务提供比随机初始化更好的 inductive bias

#### 优势3：无学习 VFM 特征提升
- 利用 3DGS 的渲染能力，直接将 2D VFM 知识注入 3D 空间
- 无需微调 VFM（冻结），计算成本可控
- 多尺度融合增强从粗到细的语义一致性

### 局限性分析

#### 局限1：场景合成简单
- 场景通过平移/旋转/缩放组合单物体 3DGS 生成，缺乏真实场景的复杂遮挡和物理交互
- 未涉及真实场景的 3DGS 重建
- 物体间无物理约束（如支撑关系）

#### 局限2：LLM 大小与性能悖论
- Qwen3-8B 反而劣于 0.6B，作者未深入解释
- 可能限制未来扩展到更强 LLM 的路径
- LoRA 参数设置未针对不同大小模型调整

#### 局限3：评估序列长度
- 未分析不同序列长度下的性能变化
- 长序列中的错误累积问题未讨论

#### 局限4：无真实机器人验证
- 停留在 perception 层面，未与真实机器人操作闭环
- 与已有 VLA 模型的集成路线未探索

### 适用性与场景分析

#### 适用场景
- **长序列任务规划**：多步操作的 affordance 理解
- **复杂场景 affordance 预分析**：为 VLA 模型提供"可操作区域+顺序"的先验
- **指令-操作对齐**：将自然语言指令解析为可执行的操作序列

#### 不适用场景
- **单步简单操作**：可能过度复杂（SeqSplatNet 在单步上也是 SOTA，但架构开销大）
- **实时高频控制**：自回归 LLM 推理延迟不适合 100Hz+ 控制
- **未见物体类别**：LLM 的常识推理可部分缓解，但 affordance 定位仍依赖训练分布

## 与相关论文对比

### [[3DAffordSplat_Efficient_Affordance_Reasoning|3DAffordSplat]] - 首个 3DGS Affordance

#### 方法对比
| 对比维度 | 3DAffordSplat | SeqAffordSplat |
|----------|---------------|----------------|
| 任务范围 | 单物体、单步 | 场景级、序列 |
| 场景组成 | 单个 Gaussian 物体 | 多物体组合场景 |
| 推理方式 | 语言编码器+分类 | LLM 自回归生成序列 |
| 语言模型 | RoBERTa（~125M） | Qwen3-0.6B |
| 预训练 | CMSA 跨模态对齐 | 条件几何重建 |
| 数据集规模 | 23,677 GS 实例 | 1,800+ 场景、14,000+ mask |

#### 关系分析
- **关系类型**：扩展（extends）— SeqAffordSplat 的三场景数据直接来源于 3DAffordSplat
- **本文优势**：从单步到序列的范式升级、更强的 LLM、更丰富的语义注入
- **共同作者**：Guanbin Li 在两篇论文中均有贡献

### SeqAfford - 首个序列 Affordance（点云）

| 对比维度 | SeqAfford | SeqAffordSplat |
|----------|-----------|----------------|
| 3D 表示 | 点云 | 3DGS |
| 推理引擎 | 基于规则/模板 | LLM 自回归 |
| 定位精度 | 低（点云稀疏性） | 高（3DGS 连续表示） |
| sIoU | 12.1 | **26.2**（2.2×） |

## 技术路线定位

### 所属技术路线
本文属于 **3D Affordance Reasoning on 3DGS** 技术路线，是 3DAffordSplat → SeqAffordSplat 的"单步→序列"范式升级。

### 技术路线发展历程
```
3D PC Affordance → Language-PC Affordance → Seq PC Affordance → Single 3DGS Affordance → Seq 3DGS Affordance (本文)
  (3DAffordanceNet)    (LASO)                (SeqAfford)        (3DAffordSplat)            (SeqAffordSplat)
     2021                 2024                  2025                2025                        2025
```

### 本文在技术路线中的位置
- **承上**：基于 3DAffordSplat 的数据和 3DGS 表示，引入 SeqAfford 的序列思想
- **启下**：LLM + 3DGS affordance 的统一框架为未来的具身操作规划奠定了基础
- **关键节点**：affordance reasoning 从"what"到"in what order"的范式转折点

## 未来工作建议

### 作者未明确提出的方向
1. **真实场景 3DGS 重建**：将 SeqSplatNet 应用于真实世界扫描的 3DGS 场景
2. **与 VLA 模型闭环集成**：将 affordance 序列输出作为 π₀/Hi Robot 等 VLA 模型的 planning 前端
3. **序列长度自适应**：处理变长序列和动态重规划

### 基于分析的改进方向
1. **更大 LLM 的有效利用**：探索为什么 Qwen3-8B 性能退化，设计更好的微调策略
2. **场景物理约束建模**：在场景合成中引入物理合理性（如稳定性、支撑关系）
3. **交互式 affordance**：从 affordance 识别扩展到 affordance 执行效果的 forward prediction
4. **多模态 grounding 增强**：不仅利用 VFM 语义，还可引入 3D VFM（如 PointLLM）增强 3D 语义

## 我的综合评价

### 价值评分

#### 总体评分
**8.8/10** — 范式创新明确，方法设计优雅（⟨SEG⟩ token 机制尤为亮眼），实验全面有说服力

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | 首次定义场景级序列 3DGS affordance 任务，⟨SEG⟩ token 驱动的端到端统一框架设计优雅 |
| 技术质量 | 8.5/10 | 预训练+语义注入双重增强合理，但场景合成为简单几何变换、无物理约束 |
| 实验充分性 | 8.5/10 | 三设置评估+交叉数据集验证+逐步消融，LLM 大小反直觉发现增加了实验深度 |
| 写作质量 | 8.5/10 | 结构清晰，teaser 图信息密集，技术细节完整 |
| 实用性 | 8/10 | 为 VLA 提供潜在 planning 前端，但缺少真实世界验证，距离部署有距离 |

### 重点关注
- **⟨SEG⟩ token 机制**：这是本文最值得学习的核心设计——将多模态生成嵌入到 LLM 自回归框架中的方法可推广到许多领域
- **LLM 大小悖论**：Qwen3-0.6B > Qwen3-8B，这对 VLA 领域的 LLM 选择有启发意义——不是越大越好
- **DINOv2 > CLIP**：局部空间语义对 affordance 定位比全局语义更有价值，这对 VFM 选择有指导意义

## 我的笔记

%%
SeqAffordSplat 是 3DAffordSplat 的自然扩展，但扩展方向非常精准——从 single-object single-step 到 scene-level sequential。这不是简单的"数据更多、场景更复杂"，而是任务范式的根本转变。

⟨SEG⟩ token 的设计是这个框架的灵魂。它巧妙地将序列长度可变性问题转化为 LLM 自回归生成长度问题，无需预设最大步数、无需显式层次分解。这种"让 LLM 决定何时输出 mask"的思路非常优雅。

从具身 AI 整体视角看，SeqAffordSplat 填补了"指令理解→动作执行"之间的关键 gap——"操作规划"（affordance-level planning）。未来将其作为 VLA 的 planning 前端，由 VLA 执行每个原子 affordance 对应的具体动作，可能是一个有前途的方向。

Qwen3-8B 性能退化这个反直觉发现值得关注。作者用 LoRA rank=8 对所有模型统一处理——对大模型可能 rank 不够。但这也暗示，在 embodied AI 任务中，中等大小的 LLM 可能是更优选择。
%%

## 相关论文

### 直接相关
- [[3DAffordSplat_Efficient_Affordance_Reasoning|3DAffordSplat]] - 首个 3DGS affordance 数据集+模型（数据基础，共同作者）
- SeqAfford - 首个序列 affordance（点云，任务范式先导）
- LASO - 语言驱动点云 affordance 分割

### 背景相关
- [[pi0_Vision-Language-Action_Flow_Model|π₀]] - 可集成 affordance 序列作为 planning 前端
- [[Hi_Robot_Hierarchical_VLA|Hi Robot]] - 层级 VLA，System 2 可从 affordance 序列受益
- DINOv2/CLIP - 2D VFM 语义特征提取

### 后续工作
- 真实场景 3DGS 扫描 + affordance reasoning
- Affordance-sequence VLA（SeqAffordSplat + π₀/Hi Robot）

## 外部资源
- --

> [!tip] 关键启示
> ⟨SEG⟩ token 驱动的序列生成范式优雅地统一了 LLM 推理和 3D 感知——"让 LLM 自回归决定何时生成 mask"而非"先规划后执行"。这一设计范式可推广到任意"序列多模态输出"任务。

> [!warning] 注意事项
> - 场景为合成组合（平移+旋转+缩放），非真实 3DGS 重建场景
> - Qwen3-8B 性能反而不如 0.6B，更大 LLM 不一定更好
> - 序列错误累积和长序列性能衰减未被充分研究
> - 定位在 perception 层面，尚未与真实机器人操作闭环

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！如果你关注 affordance reasoning、VLA planning、或多模态 LLM 应用，这是必读论文。⟨SEG⟩ token 的设计值得所有做"序列多模态输出"方向的研究者学习。
