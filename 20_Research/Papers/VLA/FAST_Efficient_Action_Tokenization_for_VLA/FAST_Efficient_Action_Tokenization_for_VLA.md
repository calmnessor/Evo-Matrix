---
date: "2025-01-17"
paper_id: "arXiv:2501.09747"
title: "FAST: Efficient Action Tokenization for Vision-Language-Action Models"
authors: "Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, Sergey Levine"
domain: "VLA"
tags:
  - 论文笔记
  - VLA
  - Action-Tokenization
  - DCT
  - BPE
  - Robot-Learning
  - Autoregressive-Policy
  - π0
quality_score: "8.5/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# FAST: Efficient Action Tokenization for Vision-Language-Action Models

## 核心信息
- **论文ID**：arXiv:2501.09747
- **作者**：Karl Pertsch*, Kyle Stachowicz*, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, Sergey Levine
- **机构**：Physical Intelligence, UC Berkeley, Stanford (* 核心贡献者)
- **发布时间**：2025-01-17
- **会议/期刊**：--
- **链接**：[arXiv](https://arxiv.org/abs/2501.09747) | [PDF](https://arxiv.org/pdf/2501.09747) | [项目页面](https://pi.website/research/fast)
- **代码**：HuggingFace `physical-intelligence/fast`

## 摘要翻译

### 英文摘要
Autoregressive sequence models, such as Transformer-based vision-language action (VLA) policies, can be tremendously effective for capturing complex and generalizable robotic behaviors. However, such models require us to choose a tokenization of our continuous action signals, which determines how the discrete symbols predicted by the model map to continuous robot actions. We find that current approaches for robot action tokenization, based on simple per-dimension, per-timestep binning schemes, typically perform poorly when learning dexterous skills from high-frequency robot data. To address this challenge, we propose a new compression-based tokenization scheme for robot actions, based on the discrete cosine transform. Our tokenization approach, FAST (Frequency-space Action Sequence Tokenization), enables us to train autoregressive VLAs for highly dexterous and high-frequency tasks where standard discretization methods fail completely. Based on FAST, we release FAST+, a universal robot action tokenizer, trained on 1M real robot action trajectories. It can be used as a black-box tokenizer for a wide range of robot action sequences, with diverse action spaces and control frequencies. Finally, we show that, when combined with the π0 VLA, our method can scale to training on 10k hours of robot data and match the performance of diffusion VLAs, while reducing training time by up to 5x.

### 中文翻译
基于 Transformer 的自回归 VLA（视觉-语言-动作）策略在捕捉复杂且可泛化的机器人行为方面非常有效。然而，这类模型需要对连续动作信号进行 tokenization，这决定了模型预测的离散符号如何映射为连续机器人动作。我们发现，当前基于简单的逐维度、逐时间步分箱方案的机器人动作 tokenization 方法，在学习高频机器人数据的灵巧技能时表现很差。为解决这一挑战，我们提出了一种基于离散余弦变换的全新压缩式机器人动作 tokenization 方案。我们的 tokenization 方法 FAST（频域动作序列 tokenization），使我们能够在标准离散化方法完全失败的高灵巧度、高频任务上训练自回归 VLA。基于 FAST，我们发布了 FAST+，一个通用机器人动作 tokenizer，在 100 万条真实机器人动作轨迹上训练，可作为黑盒 tokenizer 广泛用于各种动作空间和控制频率的机器人动作序列。最后，我们展示了当与 π0 VLA 结合时，我们的方法可扩展到 10k 小时机器人数据的训练，并匹配扩散 VLA 的性能，同时将训练时间减少多达 5 倍。

### 核心要点提炼
- **研究背景**：VLA 模型需要将连续动作离散化为 token，现有方法使用简单的逐维分箱方案
- **研究动机**：naive tokenization 在高频控制数据上表现极差，因为相邻时间步高度相关，导致模型陷入 trivial 解（直接复制上一帧动作）
- **核心方法**：使用离散余弦变换（DCT）将动作序列从时域转换到频域，压缩高频冗余信息，再用 BPE 进一步压缩
- **主要结果**：在 7 个评估环境上超越 naive tokenization，π0-FAST 匹配扩散 VLA 性能且训练快 5x，首次实现 DROID 数据集的 zero-shot 泛化
- **研究意义**：为自回归 VLA 提供了高效的动作 tokenization 方案，缩小了自回归与扩散 VLA 之间的差距

## 研究背景与动机

### 领域现状
当前机器人策略模型的主流范式是将视觉-语言模型（VLM）微调为 VLA 模型，直接输出底层机器人控制指令。这些模型需要将连续的动作信号转换为离散 token。最常用的方法是简单的逐维度、逐时间步的分箱方案（如 RT-2、OpenVLA 使用的方法），将每个动作维度独立离散化为 256 个 bins。

### 现有方法的局限性
1. **高频数据下的性能崩溃**：当控制频率升高时（如 50Hz），naive tokenization 产生的 token 数量急剧增加（如双手 50Hz 产生 700 个 token），训练困难且推理极慢
2. **时间相关性导致学习信号弱**：高频动作序列中相邻时间步高度相关，边际信息量趋近于零，模型只需"复制上一帧"即可获得低 loss，陷入局部最优
3. **无法扩展到大模型预训练**：OpenVLA 在低频 BridgeV2 上表现良好，但无法拟合高频 DROID 数据集

### 研究动机
- 动作 tokenization 是自回归 VLA 的核心瓶颈，需要一种能压缩高频冗余信息的 tokenization 方法
- 类比 NLP 中的 BPE 压缩文本、JPEG 中的 DCT 压缩图像，机器人动作也需要对应的压缩式 tokenization

## 研究问题

### 核心研究问题
如何设计一种高效的机器人动作 tokenization 方案，使得自回归 VLA 能够在高频率、高灵巧度的机器人控制任务上有效训练？

## 方法概述

### 核心思想
将动作序列视为连续时间序列信号，使用离散余弦变换（DCT）将其从时域转换到频域，保留低频（重要）系数、丢弃高频（冗余）分量，实现有损压缩；再使用 BPE 对稀疏的 DCT 系数矩阵进行无损压缩，最终得到紧凑、高信息密度的 token 序列。

### 方法框架

#### 整体架构

![[dct_method.pdf|800]]

> 图1：FAST 动作 tokenization 流水线。给定归一化后的动作块，先应用 DCT 将信号转换到频域，然后量化 DCT 系数，最后使用 BPE 将展开的逐维 DCT 系数压缩为最终的动作 token 序列。

FAST tokenization 包含以下步骤：

1. **归一化**：将训练数据集中每个动作维度的 1st 和 99th 分位数映射到 $[-1, 1]$
2. **DCT 变换**：对每个动作维度分别应用离散余弦变换，将时域信号转换为频域系数
3. **量化压缩**：通过 scale-and-round 操作 $\bar C^i_j = \text{round}(\gamma \cdot C^i_j)$ 丢弃不重要的高频系数
4. **BPE 压缩**：将量化后的稀疏 DCT 系数矩阵按列优先（低频优先）展平，训练 BPE tokenizer 进行无损压缩

#### 各模块详细说明

**模块1：归一化**
- **功能**：将原始动作数据映射到固定范围
- **输入**：原始动作序列 $a_{1:H}$
- **输出**：归一化后的动作序列，范围 $[-1, 1]$
- **关键技术**：使用分位数归一化（而非 min-max），对异常值鲁棒

**模块2：DCT 变换**
- **功能**：将时域动作信号转换到频域，集中信息到少数低频系数
- **关键技术**：DCT 将信号表示为不同频率余弦波的和，低频系数捕捉信号整体形状，高频系数反映突变
- **数学公式**：对每个动作维度 $i$ 应用 DCT 得到系数矩阵 $C^i_j$

**模块3：量化压缩**
- **功能**：通过缩放和取整操作丢弃不重要的高频系数
- **超参数**：$\gamma$（缩放系数），控制压缩率与保真度的权衡
- **默认值**：$\gamma = 10$，在保真度和压缩率间取得良好平衡

**模块4：BPE 压缩**
- **功能**：将稀疏的 DCT 系数矩阵压缩为密集 token 序列
- **词汇量大小**：默认 1024
- **关键设计**：列优先展平（低频优先），使得自回归预测先预测信号的整体形状，有利于策略的稳定 rollout
- **训练**：BPE tokenizer 可在数分钟内训练完成

### FAST 算法

**Tokenization 流程**：
1. $C^i_j \gets \text{DCT}(a^i_{1:H})$ — 计算 DCT 系数
2. $\bar C^i_j \gets \text{round}(\gamma \cdot C^i_j)$ — 量化系数
3. $[T_k] \gets [\bar C^1_1, \bar C^2_1, \dots, C^1_2, \dots, C^n_H]$ — 展平 token（列优先）

**训练 BPE**：
- 在训练集上训练 BPE tokenizer，合并频繁出现的系数组合

**推理时**：
- 所有操作均可逆，可快速解码预测的动作

## 实验结果

### 实验目标
验证 FAST tokenization 在多种 VLA 架构上的有效性，与 naive tokenization 和 VQ-based tokenization（FSQ）对比，并测试与扩散 VLA 的性能差距。

### 数据集

| 数据集 | 动作维度 | 控制频率 | 描述 |
|--------|----------|----------|------|
| BridgeV2 | 7 | 5 Hz | 单臂桌面操作 |
| DROID | 7 | 15 Hz | 大规模多任务"in-the-wild"操作 |
| Table Bussing | 7 | 20 Hz | 单臂清理桌面，12 个物体分类 |
| T-Shirt Folding | 14 | 50 Hz | 双臂折叠 T 恤 |
| Libero | 7 | -- | 仿真基准（Spatial/Object/Goal/10）|
| Grocery Bagging | 7 | 20 Hz | 单臂将物品装入购物袋 |
| Toast out of Toaster | 14 | 50 Hz | 双臂从烤面包机中取出面包 |
| Laundry Folding | 14 | 50 Hz | 双臂从洗衣篮中取出衣物并折叠 |

### 实验设置

#### 基线方法
- **Naive tokenization**：逐维逐时间步分箱，RT-2/OpenVLA 使用的方法
- **FSQ**：基于向量量化的压缩 tokenization（VQ-VAE 的简化版）
- **扩散 π0**：原始 π0 模型使用 flow-matching 扩散解码

#### 评估指标
- 各任务成功率
- 训练收敛速度
- 压缩率（token 数量对比）

#### 实验环境
- 7 个评估环境（6 个真实机器人 + 1 个仿真）
- VLA 架构：π0 (PaliGemma-3B) 和 OpenVLA (Prismatic-7B)

### 主要结果

#### 压缩率对比

| 数据集 | 控制频率 | Naive Tokens | FAST Tokens | 压缩率 |
|--------|----------|-------------|-------------|--------|
| BridgeV2 | 5 Hz | 35 | 20 | 1.75x |
| DROID | 15 Hz | 105 | 29 | 3.6x |
| Table Bussing | 20 Hz | 140 | 28 | 5.0x |
| T-Shirt Folding | 50 Hz | 700 | 53 | 13.2x |

> 关键发现：FAST 始终产生约 30 个 token/手臂/秒，与频率几乎无关，说明它真正捕捉了动作信号的底层复杂度。

![[main_result.pdf|800]]

> 图2：不同 tokenization 方法的策略性能对比。FAST 和 FSQ 等压缩式 tokenization 显著优于 naive tokenization，尤其在灵巧高频任务上。FAST 在真实机器人灵巧任务上优于 FSQ。FAST+（通用 tokenizer）匹配数据集专属 tokenizer 的性能。

#### 关键结果
1. **Naive tokenization 在 20Hz+ 任务上完全失败**：Table Bussing (20Hz) 和 T-Shirt Folding (50Hz) 上无法取得任何进展
2. **FAST 优于 FSQ**：尽管 FSQ 需要额外的神经网络训练，FAST 更简单且性能更好
3. **首次在 DROID 上实现 zero-shot 泛化**：FAST 使 DROID 策略首次能在完全未见的环境中 zero-shot 评估
4. **FAST+ 通用 tokenizer 匹配专属 tokenizer**：在未见过的机器人形态上仍保持良好压缩效果

![[droid_quali.pdf|800]]

> 图3：FAST 训练的 DROID 策略在三个大学校园中的 zero-shot 评估环境。同一 checkpoint 可在不同场景中泛化执行各种桌面操作任务。

#### 与扩散 VLA 对比

![[pi0_single_task.pdf|800]]

> 图4：扩散 π0 与 FAST π0 在单任务训练上的对比。小数据集上相当，大数据集上 FAST 收敛更快。在 DROID 上，FAST 更好地遵循语言指令。

![[pi0_multi_task.pdf|800]]

> 图5：π0-FAST 通用策略与扩散 π0 的对比。π0-FAST 匹配扩散 π0 的性能，同时训练时间减少 5x。

![[convergence_2.jpg|800]]

> 图6：训练收敛速度。π0-FAST 达到高性能所需的 GPU 小时数远少于扩散 π0（约 5x 加速）。

### 消融实验

**1. VLA 架构独立性**

![[openvla_results.pdf|500]]

> 在 OpenVLA (Prismatic-7B) 上使用 FAST+ tokenizer 训练高频 T-Shirt Folding 任务，显著提升了原始 OpenVLA 的性能，证明 FAST 不依赖于特定模型架构。

**2. BPE 步骤的重要性**

![[nobpe_results.pdf|500]]

> 去掉 BPE 步骤后性能下降：DCT 变换仍然能集中信息到少数 token，但没有 BPE 压缩时会产生大量重复的零 token，稀释学习信号且显著降低推理速度。

**3. 压缩-保真度权衡**

![[fig_dataset_comparison.pdf|500]]

> FAST 在高保真度范围内表现优异，虽然低精度范围内不如 VQ，但精细控制领域恰好最需要高保真度。

## 深度分析

### 研究价值评估

#### 理论贡献
- **贡献**：将 DCT + BPE 的经典压缩思路系统性地引入机器人动作 tokenization
  - 创新点：首次从"时间序列压缩"的角度分析动作 tokenization 问题，揭示了高频数据中 token 相关性与学习效率的关系
  - 学术价值：为自回归 VLA 的动作 tokenization 提供了新的理论基础
  - 影响范围：VLA 模型设计、机器人学习、序列建模

#### 实际应用价值
- **应用场景**：任何使用自回归模型进行机器人控制的系统
  - 优势：极简单（仅 2 个超参数），兼容任何预训练自回归 Transformer，无需修改模型架构
  - 潜在影响：降低自回归 VLA 的训练成本，使其成为扩散 VLA 的有力替代方案
- **开源 FAST+**：HuggingFace AutoProcessor 接口，3 行代码即可使用

#### 领域影响
- **短期**：使自回归 VLA 能够在高频灵巧任务上有效训练
- **中期**：可能缩小甚至消除自回归与扩散 VLA 的性能差距
- **长期**：统一动作 tokenization 标准，类似 NLP 中的 BPE 之于文本

### 方法优势详解

1. **极简设计**：仅 2 个超参数（γ 和 BPE 词汇量），无需训练神经网络，避免 VQ-VAE 的超参数敏感和训练不稳定问题
2. **高压缩率**：在高频数据上可达 10x+ 的压缩，token 数量与信号复杂度相关而非采样率
3. **架构无关**：可直接应用于任何预训练自回归 Transformer
4. **可逆解码**：所有操作均可逆，推理时快速解码

### 局限性分析

1. **推理速度较慢**：π0-FAST 需要 750ms 推理 1 秒动作块（vs 扩散 π0 的 100ms），因为需要自回归解码 30-60 个 token
   - 原因：需使用完整的 2B 参数 LLM backbone 进行自回归解码
   - 可能的解决方案：推测解码、量化、自定义推理 kernel
2. **仅在静态操作任务上验证**：所有评估任务都是静态桌面操作，未在动态任务或移动机器人上测试策略性能
3. **非完全无损压缩**：DCT 量化引入的有损压缩在某些精密操作中可能有影响
4. **未探索移动/类人机器人**：FAST+ 在离线实验中展示了压缩能力，但未在这些平台上进行实际策略评估

### 适用性与场景分析

**适用场景**：
- 高频灵巧操作任务（>20Hz）
- 大规模多任务机器人数据训练
- 需要降低自回归 VLA 训练成本的场景
- 跨机器人形态的动作 tokenization

**不适用场景**：
- 极低延迟需求（如高速动态任务）
- 已有成熟的扩散 VLA 流水线且训练成本可接受

## 与相关论文对比

### [[π0 - A Vision-Language-Action Flow Model for General Robot Control]]
- **关系类型**：直接构建在其之上
- **本文改进**：将 π0 的扩散解码替换为 FAST + 自回归解码，提升训练效率 5x
- **性能**：匹配扩散 π0 的性能，且更好地遵循语言指令

### [[OpenVLA - An Open-Source Vision-Language-Action Model]]
- **关系类型**：对比基线
- **核心差异**：OpenVLA 使用 naive per-dimension binning tokenization，在 DROID 上无法有效训练
- **本文贡献**：在 OpenVLA 上使用 FAST tokenization 显著提升了其在高频任务上的性能

### [[RT-2 - Vision-Language-Action Models Transfer Web Knowledge to Robotic Control]]
- **关系类型**：继承 token 覆写策略
- **核心差异**：RT-2 使用 256-bin 离散化，仅适合低频控制
- **本文改进**：将动作 tokenization 升级为频率域压缩方案

## 技术路线定位

### 所属技术路线
本文属于 **自回归 VLA 动作 tokenization** 技术路线，核心特点是：
- 不修改预训练模型架构
- 通过更好的 tokenization 提升训练效率
- 将 NLP 中的 tokenization 经验迁移到机器人领域

### 技术路线发展历程
```
Naive Binning (RT-1/RT-2) → VQ-based 压缩 (Behavior Transformer) → FAST (DCT+BPE) → 未来：通用动作 tokenizer + 快速推理
```

### 本文在技术路线中的位置
- **承上**：继承了 RT-2 的 token 覆写策略和 π0 的 VLA 架构
- **启下**：为自回归 VLA 的高效训练开辟了新范式，FAST+ 提供了通用 tokenizer 基准

## 未来工作建议

### 作者建议的未来工作
1. 在移动机器人和类人机器人上测试策略性能
2. 探索其他压缩方案（如 Huffman coding、Lempel-Ziv）
3. 将压缩式动作编码与非自回归解码（如扩散）结合
4. 加速自回归 VLA 推理（推测解码、量化等）

### 基于分析的未来方向
1. **FAST + 扩散解码**：使用压缩后的 token 作为扩散模型的输入/输出
2. **自适应压缩率**：根据任务精度需求动态调整 γ 参数
3. **多模态 token 联合训练**：将动作 token 与视觉/语言 token 统一训练
4. **动作 token 的预训练**：类似于语言模型的 token embedding 预训练

## 我的综合评价

### 价值评分

#### 总体评分
**8.5/10** - 方法简洁优雅，实验充分全面，解决了自回归 VLA 的关键瓶颈，具有重要的实用价值

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 将 DCT+BPE 经典压缩流程引入动作 tokenization，视角新颖但组件本身非首创 |
| 技术质量 | 9/10 | 方法严谨，理论分析清晰（通过 toy example 阐明 token 相关性对学习的影响） |
| 实验充分性 | 9/10 | 7 个评估环境、多种 VLA 架构、详细消融、与扩散 VLA 全面对比 |
| 写作质量 | 9/10 | 结构清晰，图表丰富，toy example 帮助理解核心思想 |
| 实用性 | 8/10 | FAST+ 已开源，接口友好，推理速度是当前主要限制 |

### 重点关注

- DCT 变换在时间序列压缩中的有效性
- BPE 压缩对 token 信息密度的提升
- 通用 tokenizer FAST+ 的跨机器人形态泛化能力
- 自回归 vs 扩散 VLA 的架构权衡

## 我的笔记

%% 用户可在此添加个人阅读笔记 %%

核心启发：
1. 动作 tokenization 不是简单的离散化问题，而是时间序列压缩问题
2. 高频 = 高冗余 → 压缩是必要的
3. DCT + BPE = 频域压缩 + 统计压缩，两者互补
4. 通用 tokenizer 的概念类似于 NLP 中的通用文本 tokenizer

与我的研究方向关联：
- 在 VLA 训练中，动作 tokenization 是不可忽视的一环
- FAST 可作为自回归 VLA 训练的基础组件
- 未来可探索将 FAST 与其他 VLA 架构（如我的项目中使用的模型）结合

## 相关论文

### 直接相关
- [[π0]] - 直接构建在 π0 VLA 之上
- [[OpenVLA]] - 对比基线，也在 OpenVLA 上验证
- [[RT-2]] - 继承了 token 覆写策略

### 背景相关
- [[Diffusion Policy]] - 动作 chunking 的来源
- [[DROID]] - 使用 DROID 数据集训练和评估
- [[ALOHA]] - 双臂操作任务设置来源

### 后续工作
- FAST+ tokenizer 已在 HuggingFace 发布，可被任何自回归 VLA 使用

## 外部资源
- 项目页面：https://pi.website/research/fast
- HuggingFace 模型：`physical-intelligence/fast`
- 代码使用示例见论文 Method 部分

> [!tip] 关键启示
> 动作 tokenization 的核心不是离散化精度，而是信息压缩——通过 DCT 将高频冗余信号压缩为少量高信息密度 token，是自回归 VLA 成功训练的关键。

> [!warning] 注意事项
> - FAST 不是完全无损的，需要在压缩率和保真度之间权衡
> - 当前推理速度（750ms/chunk）仍远慢于扩散方案（100ms/chunk）
> - FAST+ 虽声称"通用"，但在未见过的极端机器人形态上仍需验证

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐阅读！这是解决自回归 VLA 动作 tokenization 问题的里程碑式工作，方法简洁、实验扎实、实用性强，且已开源。
