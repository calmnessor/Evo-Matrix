---
date: "2026-05-18"
paper_id: "arXiv:2605.15195"
title: "VGGT-Ω: Scaling Feed-Forward Reconstruction Models"
authors: "Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, Christian Rupprecht"
domain: "多模态技术"
tags:
  - 论文笔记
  - 多模态技术
  - 3D-Reconstruction
  - Feed-Forward-Reconstruction
  - ViT
  - Dynamic-Scenes
  - Self-Supervised-Learning
  - Register-Attention
  - VLA
  - Language-Alignment
  - VGGT
  - DINOv3
  - CVPR2026
quality_score: "9.2/10"
created: "2026-05-18"
updated: "2026-05-18"
status: analyzed
---

# VGGT-Ω: Scaling Feed-Forward Reconstruction Models

## 核心信息
- **论文ID**：arXiv:2605.15195
- **作者**：Jianyuan Wang (Oxford), Minghao Chen (Oxford), Shangzhan Zhang (Oxford), Nikita Karaev (Oxford), Johannes Schönberger (Meta AI), Patrick Labatut (Meta AI), Piotr Bojanowski (Meta AI), David Novotny (Meta AI), Andrea Vedaldi (Oxford & Meta AI), Christian Rupprecht (Oxford)
- **机构**：Visual Geometry Group (VGG), University of Oxford & Meta AI
- **发布时间**：2026-05-14
- **会议/期刊**：CVPR 2026
- **链接**：[arXiv](https://arxiv.org/abs/2605.15195) | [Project Page](http://vggt-omega.github.io/)
- **引用**：--

## 摘要翻译

### 英文摘要
Recent feed-forward reconstruction models, such as VGGT, have proven competitive with traditional optimization-based reconstructors while also providing geometry-aware features useful for other tasks. Here, we show that the quality of these models scales predictably with model and data size. We do so by introducing VGGT-Ω, which substantially improves reconstruction accuracy, efficiency, and capabilities for both static and dynamic scenes. To enable training this model at an unprecedented scale, we introduce architectural changes that improve training efficiency, a high-quality data annotation pipeline that supports dynamic scenes, and a self-supervised learning protocol. We simplify VGGT's architecture by using a single dense prediction head with multi-task supervision and removing the expensive high-resolution convolutional layers. We also use registers to aggregate scene information into a compact representation and introduce register attention, which restricts inter-frame information exchange to these registers, in part replacing global attention. In this way, during training, VGGT-Ω uses only ~30% of the GPU memory of its predecessor, allowing us to train with 15× more supervised data than prior work and to leverage vast amounts of unlabeled video data. VGGT-Ω achieves strong results for reconstruction of static and dynamic scenes across multiple benchmarks, for example, improving over the previous best camera estimation accuracy on Sintel by 77%. We also show that the learned registers can improve vision-language-action models and support alignment with language, suggesting that reconstruction can be a powerful and scalable proxy task for spatial understanding.

### 中文翻译
近期的前馈重建模型（如 VGGT）已被证明能与传统的基于优化的重建方法竞争，同时还提供了可用于其他任务的几何感知特征。本文我们展示了这些模型的质量随模型规模和数据规模的增大而可预测地提升。我们通过引入 VGGT-Ω 来实现这一点，该模型在静态和动态场景的重建精度、效率和能力方面都有显著提升。为了实现前所未有的训练规模，我们引入了提高训练效率的架构改进、支持动态场景的高质量数据标注流水线，以及自监督学习协议。我们通过使用单一密集预测头与多任务监督、移除昂贵的高分辨率卷积层来简化 VGGT 架构。我们还使用 registers（寄存器标记）将场景信息聚合成紧凑表示，并引入 register attention（寄存器注意力），将帧间信息交换限制在这些寄存器上，部分替代全局注意力。通过这种方式，在训练期间，VGGT-Ω 仅使用其前身约 30% 的 GPU 内存，使得我们可以使用比先前工作多 15 倍的监督数据和大量无标注视频数据进行训练。VGGT-Ω 在多个基准测试中取得了静态和动态场景重建的优异结果，例如在 Sintel 上相机估计精度相比此前最佳结果提升了 77%。我们还展示了学习到的寄存器可以改进视觉-语言-动作模型并支持与语言的对齐，表明重建可以成为空间理解的强大且可扩展的代理任务。

### 核心要点提炼
- **研究背景**：前馈3D重建模型已展现出匹敌甚至超越传统SfM的性能，但其扩展性尚未被充分探索
- **研究动机**：探索前馈重建模型是否可以通过扩展数据规模和模型容量获得可预测的性能提升
- **核心方法**：引入 Register Attention 替代部分全局注意力、单密集预测头+多任务损失、大规模动态视频标注流水线、以及自监督学习协议
- **主要结果**：在 6 个基准（3静态+3动态）上全面超越 SOTA，Sintel 相机估计 AUC@3° 从 22.5 跃升至 40.0（提升 77%），深度 δ1.25 从 86.1 提升至 93.5
- **研究意义**：证明了前馈重建模型的 scaling law，并展示了重建可作为学习通用空间表示的强大代理任务

## 研究背景与动机

### 领域现状
3D 重建领域正经历从传统优化方法（COLMAP、Bundle Adjustment）向前馈神经网络方法的范式转变。VGGT 代表性地展示了纯前馈推理可以超越后优化方法。然而，与语言模型领域对 scaling law 的深入理解不同，3D 视觉中这种规模化特性几乎没有被探索过。

### 现有方法的局限性
1. **VGGT 的内存瓶颈**：多密集预测头（点图、追踪特征）和高分辨率 DPT 卷积层消耗大量 GPU 内存
2. **全局注意力的计算成本**：随 token 数量平方增长，限制多帧处理
3. **数据规模受限**：缺乏大规模、高质量的动态场景标注数据
4. **无自监督能力**：无法利用海量无标注视频数据

### 研究动机
探索前馈重建模型是否可以像 LLM 一样通过扩展数据和模型规模来获得可预测的性能提升，并构建一个真正可扩展的基础重建模型。

## 研究方法

### 核心思想
将重建模型的训练成本降低 70%，以此换取 15 倍以上的数据扩展和模型扩展空间，同时通过 register attention 机制让模型学习更具语义性的场景表征。

### 方法框架

#### 整体架构

![[architecture_v8.pdf|800]]

> 图1：VGGT-Ω 架构图。每个输入帧通过 DINOv3 ViT 编码为图像 token，附加相机 token 和寄存器 token（scene tokens）。交替注意力的 block 中包含全局/寄存器注意力层和帧内注意力层，最终通过单深度头预测深度图，通过相机头预测相机参数。

#### 各模块详细说明

**模块1：特征提取与Token化**
- **功能**：将输入图像编码为视觉 token
- **输入**：$N$ 张 RGB 图像 $I_1,...,I_N \in \mathbb{R}^{3\times H\times W}$
- **输出**：每帧 token $\bz_i = (\bz_i^F, \zcamera_i, \zscene_i)$，其中 $\bz_i^F \in \mathbb{R}^{H'W'\times C}$ 是 patch token, $\zcamera_i \in \mathbb{R}^{1\times C}$ 是相机 token, $\zscene_i \in \mathbb{R}^{16\times C}$ 是 16 个寄存器 token
- **处理流程**：
  1. DINOv3 初始化的 ViT 将每帧编码为 patch token（patch size=16，比 VGGT 的 14 减少约 25% token 数）
  2. 每个图像附加 1 个相机 token 和 16 个可学习的寄存器 token
  3. 这些 token 的初始值根据该帧是否为参考帧取不同的可学习参数
- **关键技术**：DINOv3 预训练权重提供强初始化，大幅加速收敛（比从头训练快 4-8×）

**模块2：Register Attention（核心创新）**
- **功能**：聚合多帧场景信息，减少计算量
- **输入**：所有帧的全部 token
- **输出**：更新后的寄存器 token（聚合了跨帧场景信息）
- **处理流程**：
  1. 在 25% 的全局注意力层中，仅寄存器 token 参与跨帧自注意力
  2. 更新后的寄存器在后续帧内注意力层中与同帧的图像 token 交互，分发聚合的信息
  3. 形成"信息瓶颈"：寄存器→聚合场景信息→分发给各帧图像 token
- **数学公式**：
   - 全局注意力：$\bz' = \operatorname{attn}(\bz)$
   - 帧内注意力：$\bz' = \operatorname{attn}_f(\bz) = (\operatorname{attn}(\bz_1), ..., \operatorname{attn}(\bz_N))$
   - 寄存器注意力：$({\zscene_1}',...,{\zscene_N}') = \operatorname{attn}(\zscene_1,...,\zscene_N)$

**模块3：轻量化解码头**
- **深度头**：保留 DPT 低分辨率卷积层 + MLP + pixel-shuffle 上采样。输出深度图和置信度两个通道。
  $$u=4,\ \text{MLP output dims:}\ (H'\times W', 2u^2) \xrightarrow{\text{pixel-shuffle}} (uH')\times(uW')\times 2$$
- **相机头**：轻量 Transformer 处理所有相机 token 和寄存器 + MLP，单次前向预测旋转四元数 $\bq_i \in \mathbb{R}^4$、平移向量 $\bt_i \in \mathbb{R}^3$ 和 FOV $\bbf_i \in \mathbb{R}^2$

**模块4：多任务损失（单头多监督）**
$$\mathcal{L} = \lambda_{\text{cam}} \mathcal{L}_{\text{cam}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{point}} \mathcal{L}_{\text{point}} + \lambda_{\text{match}} \mathcal{L}_{\text{match}}$$
- 相机损失：$\ell_1$（比 Huber 更稳定）
- 深度损失：aleatoric uncertainty + 梯度一致性
- 点图损失：深度反投影 + 与 GT 点图比较
- 匹配损失：对最后一层 token 进行特征匹配对比学习

**模块5：自监督学习**
- Teacher-Student 框架，类似 DINO
- 初始化自监督 VGGT-Ω 检查点
- 不同增强（颜色抖动、随机旋转、masking、帧重排序）应用于 teacher 和 student
- Student 匹配 teacher 的预测和特征分布
- Teacher 通过 EMA 更新（$m=0.999$）
- 深度和相机头在自监督阶段冻结

### 标注流水线架构

论文构建了大规模视频标注流水线，从约 4000 万互联网视频中筛选出 80 万高质量标注序列：

```
4000万互联网视频
   ↓ VLM预过滤（丢弃剪辑、水印、严重模糊...）
约400万可重建视频
   ↓ 动态Mask提取（Grounding DINO检测可移动物体）
   ↓ 特征匹配与追踪（SIFT + SuperPoint/SuperGlue + ALIKED/LightGlue + VGGSfM Tracker）
   ↓ 重建与过滤（VGGT初始化 + COLMAP BA + 启发式过滤）
   ↓ 多视图一致性验证（深度反投影到其他视角验证）
   ↓ 监督几何过滤（XGBoost + Random Forest + CatBoost 集成分类器）
80万高质量标注（~20万动态 + ~60万静态）
```

加上已有的公开数据集（约 300 万序列），总计约 400 万训练序列，是 VGGT 的 15 倍以上。

## 实验结果

### 实验目标
验证 VGGT-Ω 在静态和动态场景下的重建性能、训练效率提升、扩展性，以及寄存器在下游任务中的价值。

### 数据集

| 数据集 | 类型 | 评估指标 |
|--------|------|----------|
| 7 Scenes | 静态室内 | AUC@3°/30°, δ1.25, AbsRel |
| NRGBD | 静态室内 | AUC@3°/30°, δ1.25, AbsRel |
| ETH3D | 静态室内外 | AUC@3°/30°, δ1.25, AbsRel |
| DyCheck | 动态多物体 | AUC@3°/30°, δ1.25, AbsRel |
| Sintel | 动态电影 | AUC@3°/30°, δ1.25, AbsRel |
| TUM-Dynamic | 动态室内 | AUC@3°/30°, δ1.25, AbsRel |

### 实验设置

#### 基线方法
- **前馈方法**：VGGT, PI3, Depth Anything 3 (Giant-1B), MapAnything, MonST3R
- **优化方法**：MegaSaM（动态优化）
- **多尺寸**：VGGT-Ω 提供 200M / 500M / 1B / 10B 四个规模

#### 模型规格
| 规模 | 参数 | Attention Block数 | Hidden Size |
|------|------|---------------------|-------------|
| Ω-200M | 200M | 12 | 384 |
| Ω-500M | 500M | 12 | 768 |
| Ω-1B | 1B | 24 | 1024 |
| Ω-10B | 10B | 16 | 4096 |

#### 训练配置
- 优化器：AdamW，240K 迭代（160K 监督 + 50K 自监督 + 30K 最终监督）
- 学习率：峰值 $2\times10^{-4}$（监督）、$1\times10^{-4}$（自监督），warmup 5% + cosine decay
- 硬件：128 × 96GB H100 GPU, bfloat16, FSDP
- 帧数：均匀采样 [1, 24] 帧

### 主要结果

#### 相机姿态估计

![[architecture_v8.pdf|800]]

| 方法 | 7Scenes AUC@3° | NRGBD AUC@3° | ETH3D AUC@3° | DyCheck AUC@3° | Sintel AUC@3° | TUM-Dyn AUC@3° |
|------|----------------|--------------|--------------|----------------|---------------|----------------|
| VGGT | 10.9 | 81.7 | 18.8 | 21.0 | 15.0 | 16.6 |
| PI3 | 13.3 | 83.8 | 35.3 | 23.3 | 14.8 | 16.1 |
| MegaSaM | 10.6 | 17.2 | 5.9 | 26.8 | 22.5 | 15.4 |
| DA3 | 18.7 | 86.4 | 46.1 | 32.1 | 16.2 | 20.8 |
| **Ω-1B** | **29.6** | **89.7** | **49.8** | **38.4** | **35.3** | **30.2** |
| **Ω-10B** | **36.4** | **92.5** | **56.3** | **43.7** | **40.0** | **36.4** |

> Sintel AUC@3° 从 22.5 → 40.0，提升 77%；ETH3D AUC@30° 从 87.0 → 90.4

#### 深度估计

| 方法 | 7Scenes δ1.25 | NRGBD δ1.25 | ETH3D δ1.25 | DyCheck δ1.25 | Sintel δ1.25 | TUM-Dyn δ1.25 |
|------|---------------|-------------|-------------|---------------|--------------|---------------|
| VGGT | 91.9 | 99.1 | 97.4 | 95.2 | 79.2 | 92.2 |
| DA3 | 93.0 | 99.5 | 99.6 | 97.7 | 86.1 | 94.3 |
| **Ω-1B** | **94.6** | **99.6** | **99.8** | **98.4** | **89.5** | **97.4** |
| **Ω-10B** | **96.3** | **99.7** | **99.8** | **98.7** | **93.5** | **98.3** |

> Sintel δ1.25 从 86.1 → 93.5（提升 8.6%），AbsRel 从 0.118 → 0.081（降低 31%）

#### 定性结果

![[qual_v6.pdf|800]]

> 图2：VGGT-Ω 在静态和动态场景上的重建结果示例，包含交通、人体运动、自然景观和水下环境。

![[qual_comp_to_da3.pdf|800]]

> 图3：与 DA3 和 MegaSaM 在困难案例上的对比。DA3 在重复纹理（雪地缆车）和强相机翻转（无人机拍塔楼）场景中失败，而 VGGT-Ω 产生全局一致的几何。

### 消融实验

| 消融项 | 点误差 |
|--------|--------|
| 全全局注意力（无 register attention） | 0.071 |
| 25% register attention（默认） | 0.073 |
| 移除 point + matching 损失 | 0.078 |
| 加入自监督训练 | 0.070 |
| VGGT 原始多头多任务 | 0.070（但内存开销大） |

![[scale_v5_vector.pdf|800]]

> 图4：Scaling Law — 扩大模型规模（200M → 10B）和数据规模（数千 → 200万序列）均带来一致的点误差降低，呈 power-law 趋势。

### 推理效率

![[mem_speed_v2_vector.pdf|800]]

> 图5：推理内存与速度对比。VGGT-Ω 比 VGGT 快 20-25%，可处理 1250+ 帧（DA3 仅~750帧）。全 register attention 模式下处理 1000 帧仅需 11.7 秒（vs 240.2 秒），但有性能折损。

### VLA 机器人应用

| 方法 | Spatial SR | Object SR | Goal SR | Long SR | Average SR |
|------|-----------|-----------|---------|---------|------------|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **+ Frozen Scene Tokens** | **99.3** | **99.2** | **99.0** | **96.7** | **98.5** |

> 冻结 VGGT-Ω，仅将其寄存器 token 拼接至 OpenVLA-OFT 输入，在 LIBERO 全任务上提升 1.4 个点。

### 语言对齐

| 设置 | Top-1 Accuracy | Top-3 Accuracy |
|------|---------------|----------------|
| VLM 嵌入（训练目标） | 76.8% | 97.0% |
| LLM 嵌入（零样本迁移） | 47.5% | 77.8% |

> 100 个手工挑选视频上的检索实验，展示了寄存器携带高级语义信息。

## 深度分析

### 研究价值评估

#### 理论贡献
- **贡献1：前馈重建的 scaling law**
  - 创新点：首次系统证明 3D 前馈重建模型遵循 power-law 扩展规律
  - 学术价值：为 3D 基础模型研究提供了重要的理论指导
  - 影响范围：影响所有面向 3D 的大规模模型训练

- **贡献2：Register Attention 机制**
  - 创新点：将 ViT 中的 register token 从辅助角色提升为核心信息聚合机制，并引入跨帧 register 通信
  - 学术价值：为高效多帧/视频理解提供新的架构设计范式
  - 影响范围：ViT 架构设计、多帧视频理解、多模态模型

- **贡献3：重建作为空间理解代理任务**
  - 创新点：证明纯几何重建学到的表征可直接改进 VLA 和对齐语言
  - 学术价值：为"柏拉图表征假说"（Platonic Representation Hypothesis）提供实验支持
  - 影响范围：具身智能、世界模型

#### 实际应用价值
- **3D 重建**：可作为 COLMAP/SfM 的高质量替代或初始化，速度提升 50×
- **具身智能**：冻结的 register 特征可直接增强 VLA 模型
- **视频理解**：无运动监督自动发现运动区域（motion awareness 自发涌现）

#### 领域影响
- **短期影响**：成为 3D 重建社区的新 SOTA basesline，大量 follow-up 工作涌现
- **中期影响**：推动前馈重建模型替代传统 SfM pipeline
- **长期影响**：重建任务可能被整合进统一的 omni-model 中，作为多模态训练的一部分
- **潜在变革**：3D 视觉基础模型的 scaling 路径被验证，类似 DINO/CLIP 对 2D 视觉的影响

### 方法优势详解

1. **内存效率**：训练内存减少 70%，使得 128 GPU 即可训练 1B+ 模型
2. **数据效率**：15 倍数据扩展，标注流水线高效产出高质量动态数据
3. **表征质量**：寄存器 token 同时具备几何精度和语义信息
4. **扩展性验证**：从 200M 到 10B，从数千到两百万序列，性能单调提升
5. **动态场景能力**：无需显式运动模块即可处理动态内容

### 局限性分析

1. **运动模糊与极端 FOV 变化**：在强运动模糊和广角-长焦突变场景中性能下降
2. **薄结构的深度估计**：围栏等细长物体容易被背景深度"吞没"
3. **自监督提升有限**：在标准基准上自监督仅带来边际提升（0.073→0.070），主要改善 OOD 泛化
4. **全 register attention 性能损失**：完全去掉全局注意力会掉到 VGGT 级别（更适合端侧应用）
5. **深度头 artifact**：纯 MLP 解码器产生块状伪影，仍需要部分卷积层

### 适用性与场景分析

#### 适用场景
- 静态场景 3D 重建（建筑、物体）
- 动态视频 4D 重建（人体运动、交通）
- 机器人空间感知（VLA 增强）
- 通用 3D 几何特征提取

#### 不适用场景
- 超高精度度量级重建（仍需 BA 后优化）
- 极端的鱼眼/全景相机
- 具有大量镜面/透明表面的场景
- 实时 SLAM（当前推理需数秒到数十秒）

## 与相关论文对比

### VGGT - 直接前身

#### 方法对比
| 对比维度 | VGGT | VGGT-Ω |
|----------|------|--------|
| Backbone | DINOv2 (patch=14) | DINOv3 (patch=16) |
| 注意力 | 全局注意力 + 帧内注意力 | + Register Attention (25%) |
| 预测头 | 多头（点图、追踪特征、深度、相机） | 单密集头（深度）+ 相机头 |
| 监督数据 | ~25万序列 | ~400万序列 |
| 动态场景 | 有限支持 | 完整支持 |
| 训练内存 | 100% | ~30% |

#### 性能对比
- Sintel 相机 AUC@30°：50.0 → 79.1（+58%）
- 同时推理速度快 20-25%

### Depth Anything 3 (DA3) - 重要对比

| 对比维度 | DA3 | VGGT-Ω |
|----------|-----|--------|
| 核心方法 | 原生 DINO + 深度/射线图 | Register Attention + 多任务监督 |
| 最大模型 | Giant (1B) | 10B |
| 可处理帧数 | ~750 | ~1250+ |
| 推理速度 | 较慢 | 较快 |
| ETH3D AUC@30° | 87.0 | 90.4 |

### PI3

| 对比维度 | PI3 | VGGT-Ω |
|----------|-----|--------|
| 参考帧依赖 | 无（permutation-invariant） | 有（但通过 register 缓解） |
| 动态重建 | 支持 | 支持（更强） |
| Sintel AUC@30° | 53.5 | 79.1 |

## 技术路线定位

### 所属技术路线
前馈式多视角 3D 重建（Feed-Forward Multi-View 3D Reconstruction）

核心特点：
- 直接从图像端到端预测几何
- 无需逐场景优化（无 BA 迭代）
- 利用大规模数据学习几何先验

### 技术路线发展历程
```
DUSt3R (pairwise) → MASt3R (pairwise+pose) → VGGT (multi-view, all-in-one) 
   → PI3 (permutation-invariant) → DA3 (dense ray maps)
   → VGGT-Ω (scaled-up, register attention, dynamic scenes)
   → 未来：统一 omni-model 中的重建
```

### 本文在技术路线中的位置
- **承上**：继承了 VGGT 的交替注意力架构和多任务训练范式
- **启下**：建立了前馈重建的 scaling law，register attention 机制为多帧融合提供新设计空间
- **关键节点**：从"有效方法"走向"可扩展基础模型"的关键一步

## 未来工作建议

### 作者建议的未来工作
1. 纯 MLP 解码头（消除块状伪影）
2. 条件化辅助输入（只在微调阶段提供时序/相机参数）
3. 更强自监督方法的探索

### 基于分析的未来方向
1. **统一 omni-model 中的重建**：将 3D 重建作为多模态训练的一个任务，利用文字和视频的隐式 3D 知识
2. **实时推理**：全 register attention + streaming 处理实现在线场景理解
3. **跨模态对齐的深入探索**：register token 与语言的对齐展示了 3D→2D 的桥梁，或可扩展到触觉、音频等
4. **数据飞轮**：利用 VGGT-Ω 自身生成伪标签，迭代改进

## 综合评价

### 价值评分

#### 总体评分
**9.2/10** — 这是一项在 3D 基础模型方向上具有里程碑意义的工作。它不仅通过大规模的工程优化实现了显著的性能提升，更重要的是验证了前馈重建模型的 scaling law，并展示了重建作为空间理解代理任务的潜力。

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | Register Attention 巧妙地将 ViT register 从被动特征提升为主动信息聚合机制；首次系统性展示 3D 重建 scaling law |
| 技术质量 | 10/10 | 训练效率提升 70% 的工程创新扎实，标注流水线设计精良，实验严谨全面 |
| 实验充分性 | 10/10 | 6 个基准 + 4 个模型规模 + 消融 + VLA + 语言对齐 + 定性可视化 + 数据质量分析 |
| 写作质量 | 9/10 | 非常清晰，尤其是 Discussion 部分对实践经验的分享（data quality、fine-tuning tips）极有价值 |
| 实用性 | 8/10 | 模型和训练方案可复现，register token 可直接赋能多种下游任务，但训练成本（128 H100）限制了社区广泛复现 |

### 值得关注的技术点
- **Register Attention 代替全局注意力**：仅 25% 替换即可保持性能并加速
- **Data Quality 章节**：对各种数据噪声的深入分析（sensor leakage, thin structure, doming effect, fake background）是训练 3D 基础模型的宝贵经验
- **Motion Awareness 自发涌现**：无需任何运动监督，模型中间层自动分离运动区域
- **Model Souping 分析**：通过权重平均探测不同信息在模型中的存储位置
- **DINO 初始化的重要性**：加速收敛 4-8×

> [!tip] 关键启示
> 3D 基础模型的规模化路径已被验证：降低训练成本→扩展数据→扩大模型→性能可预测提升。Register 机制可能是连接几何理解和语义理解的关键桥梁。

> [!warning] 注意事项
> - 数据质量远比数量重要——论文专门用一整节讨论数据质量问题
> - 纯 MLP 解码器在室外场景会产生明显块状伪影，小型卷积仍是必要的
> - 全 register attention 虽然极快（1000帧 11.7s），但性能会退化到 VGGT 级别

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐阅读！这是 3D 基础模型方向具有里程碑意义的论文，对理解前馈重建的边界和未来方向至关重要。特别推荐 Discussion 和 Data Quality 部分。

## 相关论文

### 直接相关
- [[VGGT]] - 直接前身，VGGT-Ω 在此基础上进行架构简化和规模化
- [[DUSt3R]] - 前馈重建的奠基工作，首次将点图预测与相机估计统一
- [[Depth Anything 3]] - 同期工作，采用原生 DINO + 射线图方案
- [[PI3]] - 移除参考帧依赖的前馈重建

### 背景相关
- [[DINOv3]] - VGGT-Ω 的 ViT backbone 初始化来源
- [[Vision Transformers Need Registers]] - Register token 的原始提出
- [[MegaSaM]] - 动态重建优化方法的代表
- [[COLMAP]] - 经典 SfM pipeline

### 后续工作
- [[FastVGGT]] - 通过 token merging 加速 VGGT 推理
- [[SceneScribe-1M]] - 大规模视频标注数据集
- [[OpenVLA-OFT]] - VLA 模型中受益于几何特征增强

## 外部资源
- [Project Page](http://vggt-omega.github.io/)
- [CVPR 2026 Paper](https://arxiv.org/abs/2605.15195)
