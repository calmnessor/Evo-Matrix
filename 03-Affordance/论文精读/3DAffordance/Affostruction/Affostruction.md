---
date: "2026-05-19"
paper_id: "arXiv:2601.09211"
title: "Affostruction: 3D Affordance Grounding with Generative Reconstruction"
authors: "Chunghyun Park, Seunghyeon Lee, Minsu Cho"
domain: "Affordance_Prediction"
tags:
  - 论文笔记
  - Affordance-Prediction
  - 3D-Affordance-Grounding
  - Generative-Reconstruction
  - Flow-Matching
  - RGBD
  - Multi-View
  - Active-View-Selection
  - TRELLIS
  - Open-Vocabulary
  - CVPR2026
quality_score: "8.5/10"
created: "2026-05-19"
updated: "2026-05-19"
status: analyzed
---

# Affostruction: 3D Affordance Grounding with Generative Reconstruction

## 核心信息
- **论文ID**：arXiv:2601.09211
- **作者**：Chunghyun Park (POSTECH), Seunghyeon Lee (POSTECH), Minsu Cho (POSTECH & RLWRLD)
- **机构**：Pohang University of Science and Technology (POSTECH), RLWRLD
- **发布时间**：2026-01-14
- **会议/期刊**：CVPR 2026
- **链接**：[arXiv](https://arxiv.org/abs/2601.09211) | [Project Page](https://chrockey.github.io/Affostruction/)
- **引用**：--

## 摘要翻译

### 英文摘要
This paper addresses the problem of affordance grounding from RGBD images of an object, which aims to localize surface regions corresponding to a text query that describes an action on the object. While existing methods predict affordance regions only on visible surfaces, we propose Affostruction, a generative framework that reconstructs complete object geometry from partial RGBD observations and grounds affordances on the full shape including unobserved regions. Our approach introduces sparse voxel fusion of multi-view features for constant-complexity generative reconstruction, a flow-based formulation that captures the inherent ambiguity of affordance distributions, and an active view selection strategy guided by predicted affordances. Affostruction outperforms existing methods by large margins on challenging benchmarks, achieving 19.1 aIoU on affordance grounding and 32.67 IoU for 3D reconstruction.

### 中文翻译
本文解决从物体 RGBD 图像进行 affordance grounding 的问题，目标是定位与描述动作的文本查询相对应的表面区域。现有方法仅在可见表面上预测 affordance 区域，我们提出 Affostruction——一个生成式框架，从部分 RGBD 观测中重建完整物体几何，并在包含未观测区域的完整形状上定位 affordance。我们的方法引入了用于恒复杂度生成式重建的多视角特征稀疏体素融合、捕捉 affordance 分布固有歧义性的基于 flow 的公式化表达，以及由预测 affordance 引导的主动视角选择策略。Affostruction 在挑战性基准上大幅超越现有方法，在 affordance grounding 上达到 19.1 aIoU，在 3D 重建上达到 32.67 IoU。

### 核心要点提炼
- **研究背景**：机器人操作需要理解物体的功能属性（affordance），但现有方法仅能在观测到的表面上预测，无法处理遮挡区域
- **研究动机**：机器人从有限视角观察物体，需要同时完成几何补全和功能区域预测
- **核心方法**：基于 TRELLIS 的生成式重建 + 多视角 RGBD 稀疏体素融合 + Flow-based 生成式 affordance 预测 + affordance 驱动的主动视角选择
- **主要结果**：完整几何 affordance 19.1 aIoU（超 SOTA 40.4%），重建 32.67 IoU（超 SOTA 54.8%），部分观测 affordance 9.26 aIoU（接近两倍于两阶段方案）
- **研究意义**：首次将生成式重建与 affordance grounding 统一，实现从部分观测到完整功能理解的闭环

## 研究背景与动机

### 领域现状
3D affordance grounding 正从固定标签向开放词汇发展（OpenAD、PointRefer、Espresso-3D）。同时，3D 生成模型（TRELLIS、Shap-E、InstantMesh）和 3D 重建模型（MCC、MVSNet）各自独立发展。现有 affordance 方法假设完整点云输入，或仅在可见表面预测——两者都未解决"机器人从部分观测中推理被遮挡功能区域"的实际需求。

### 现有方法的局限性
1. **仅预测可见表面**：OpenAD、PointRefer、Espresso-3D 等直接在部分点云上预测，不补全几何
2. **生成模型不用 depth**：TRELLIS 等仅以 RGB 为输入，缺乏深度信息导致结构歧义
3. **重建模型不预测 affordance**：MCC 等仅恢复几何，不涉及功能属性
4. **无主动视角**：现有方法被动接受给定视角，不能主动选择获得更多信息的观察位置

### 研究动机
机器人通过 RGBD 相机从有限视角观察物体——例如看到杯子的正面，需要推理背面把手的抓取位置。这需要三个能力的统一：补全未观测几何、在完整形状上定位功能区域、主动选择能最大化功能信息的新视角。

## 研究方法

### 核心思想
将 affordance grounding 从"在观察到的表面上做判别式预测"转变为"先生成完整几何，再在完整形状上生成式地建模 affordance 分布"，并利用预测结果指导下一个最佳视角选择。

### 方法框架

#### 整体架构

![[overview.png|800]]

> 图1：Affostruction 整体架构。三阶段流程：(1) 多视角 RGBD 稀疏体素融合→生成式重建完整几何；(2) 基于 Flow Matching 的 affordance heatmap 生成；(3) affordance 驱动的主动视角选择。

#### 各模块详细说明

**模块1：多视角生成式重建（Stage 1）**

基于 TRELLIS 的 Flow Transformer，进行三方面扩展：

1. **稀疏体素融合（Sparse Voxel Fusion）**
   - 每个视角：DINOv2 提取特征 → 利用深度和相机参数投影到 3D 世界坐标 → 形成稀疏体素 $\mathbf{V}_i = \{(\mathbf{p}_j, \mathbf{f}_j)\}$
   - 多视角融合：重叠体素取特征平均，不重叠体素取并集 → $\bar{\mathbf{V}} = \{(\mathbf{p}_m, \bar{\mathbf{f}}_m)\}$
   - 最终 conditioning：$\mathbf{C}_{\text{voxel}} = \{\bar{\mathbf{f}}_m + \text{PE}_{\text{3D}}(\mathbf{p}_m)\}$，token 数量恒定（$O(1)$ 而非 $O(N)$）

2. **随机多视角训练**
   - 每次迭代随机采样 1-8 视角
   - 解决单视角训练的模型在多视角推理时性能退化的问题
   - 推理时性能随视角数增加单调提升，约 6-8 视角饱和

3. **Rectified Flow 训练**
   - 在 dense latent tensor $\mathbf{X} \in \mathbb{R}^{16^3 \times 8}$ (4096 tokens) 上进行 flow matching
   - $$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, \mathbf{X}_0, \epsilon}\left[||v_\theta(\mathbf{X}_t, \mathbf{C}_{\text{voxel}}, t) - (\epsilon - \mathbf{X}_0)||_2^2\right]$$

**模块2：Flow-based Affordance Grounding（Stage 2）**

- 在 Stage 1 预测的稀疏体素位置上初始化稀疏噪声张量 $\mathbf{A}_1$
- 条件化于 CLIP 文本编码器的嵌入 $\mathbf{C}_{\text{text}} = \text{CLIP}_{\text{text}}(q)$
- 使用 **binary mask loss**（BCE + Dice）替代 MSE：
  $$\mathcal{L}_{\text{mask}}(\mathbf{A}', \mathbf{A}) = \mathcal{L}_{\text{BCE}}(\mathbf{A}', \mathbf{A}) + \mathcal{L}_{\text{Dice}}(\mathbf{A}', \mathbf{A})$$
- 生成式方法天然捕捉 affordance 的多模态——同一查询（"grasp"）存在多个有效区域

**模块3：Affordance 驱动的主动视角选择**

1. **候选视角生成**：在物体周围半球上均匀采样 K=40 个候选相机姿态
2. **Affordance 可见性度量**：将 affordance mesh 渲染到候选视角，计算渲染图中 heatmap 值的总和：
   $$\mathcal{S}(\pi_i, \mathcal{M}) = \sum_{u,v} A_{\text{render}}(u,v)$$
3. **迭代选择**：选择最大化可见性评分的视角，采集新观测后重新重建和预测

## 实验结果

### 实验目标
验证 Affostruction 在三个层次上的优势：3D 重建精度、完整几何上的 affordance grounding、以及部分观测下的 affordance grounding（含主动视角选择）。

### 数据集
| 数据集 | 用途 | 规模 |
|--------|------|------|
| 3D-FUTURE + HSSD + ABO + Affogato-train | 训练重建模型 | -- |
| Affogato-train | 训练 affordance 模型 | 含 affordance 标注 |
| Toky4K | 评估重建 | 1250个物体 |
| Affogato-test | 评估 affordance | 含开放词汇查询 |

### 主要结果

#### 3D 重建（Toky4K）

| 方法 | Depth | IoU↑ | CD↓ | F-score↑ | PSNR↑ | LPIPS↓ |
|------|-------|------|-----|----------|-------|--------|
| Shap-E | -- | 6.39 | 0.672 | 0.010 | 13.07 | 0.436 |
| InstantMesh | -- | 13.68 | 0.406 | 0.039 | 16.59 | 0.299 |
| TRELLIS | -- | 19.49 | 0.369 | 0.050 | 17.61 | 0.244 |
| MCC | ✓ | 21.11 | 0.330 | 0.065 | N/A | N/A |
| **Affostruction** | ✓ | **32.67** | **0.243** | **0.100** | **18.84** | **0.192** |

> IoU 超 TRELLIS 67.6%，超 MCC 54.8%。深度信息使几何指标大幅提升。

#### 完整几何 Affordance Grounding（Affogato）

| 方法 | aIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|------|-------|------|------|------|
| OpenAD | 3.1 | 64.8 | 0.329 | 0.150 |
| PointRefer | 10.5 | 76.1 | 0.405 | 0.120 |
| Espresso-3D | 13.6 | **79.0** | **0.429** | **0.111** |
| **Affostruction** | **19.1** | 72.0 | 0.426 | 0.217 |

> aIoU 超 Espresso-3D 40.4%。虽然 AUC 略低（未微调编码器），但**空间定位精度显著更高**——这对机器人操作更重要。

#### 部分观测 Affordance Grounding（最有价值的实验）

| 方法 | 重建 | aIoU↑ | aCD↓ |
|------|------|-------|------|
| OpenAD | -- | 0.38 | 0.417 |
| PointRefer | -- | 0.53 | 0.307 |
| Espresso-3D | -- | 0.60 | 0.289 |
| TRELLIS + Espresso-3D | ✓ | 2.23 | 0.157 |
| MCC + Espresso-3D | ✓ | 4.74 | 0.135 |
| **Affostruction** | ✓ | **9.26** | **0.104** |

> 接近两倍于最佳两阶段方案（MCC+Espresso-3D: 4.74 → 9.26），证明**端到端的稀疏体素空间联合推理远比"重建→预测"解耦方案有效**。

#### 主动视角选择

![[view_sampling.png|600]]

> 从 affordance 可见性最低的初始视角开始，1 个额外视角后：affordance 驱动选择 9.2 aIoU vs 顺序 4.7（2.0× 加速），vs 随机 6.2（1.5× 加速）。到 8 视角时主动和随机收敛（13.3 vs 13.2），顺序仍落后（11.7）。

#### 效率

| 方法 | 总耗时(秒) | 峰值显存(GB) |
|------|-----------|-------------|
| TRELLIS + Espresso-3D | 8.30 | 16.90 |
| MCC + Espresso-3D | 35.37 | **4.36** |
| **Affostruction** | **7.20** | 16.35 |

> 最快推理（7.2s），得益于采样步数从 25 降到 5 步（5× 加速），且联合推理避免了解耦方案的开销。

### 消融实验

**随机多视角训练的重要性**：

![[multiview_iou.png|600]]

> 仅单视角训练的模型在多视角推理时性能退化；随机多视角训练使性能随视角数单调提升，6-8 视角饱和。稀疏体素融合优于 RGB-only patches、RGBD patches 和 MCC 特征编码。

**定性结果**：

![[partial_affordance_qual.png|800]]

> 从单个部分 RGBD 观测出发，Affostruction 能补全遮挡几何并在完整形状上精确预测 affordance heatmap。

**多物体扩展**：

![[multi_object.png|600]]

> 结合 SAM3D 分割，可将 object-centric 方法扩展到多物体场景。

**失败案例**：

![[failure_cases.png|600]]

> 两种典型失败：(1) 严重遮挡导致重建错误→传播到 affordance 预测；(2) 初始 affordance 预测错误→误导主动视角选择远离目标区域。

## 深度分析

### 研究价值评估

#### 理论贡献
- **贡献1：生成式重建 + affordance 的统一范式**
  - 创新点：首次将 3D 生成式重建与 affordance grounding 统一在稀疏体素空间中，两者共享结构和表征
  - 学术价值：提供了新的思考范式——affordance 不应是"重建后附加"，而应与几何生成深度耦合
- **贡献2：Flow-based 多模态 affordance 建模**
  - 创新点：用生成模型而非判别模型来捕捉 affordance 的固有歧义性，单一查询对应多个有效区域
  - 学术价值：为 affordance 建模提供了概率论基础
- **贡献3：Affordance 驱动的闭环视角选择**
  - 创新点：affordance→视角→更好重建→更好 affordance 的自我改进闭环
  - 学术价值：为主动感知（active perception）提供了新维度——功能驱动的视角规划

#### 实际应用价值
- **机器人操作**：从部分观测直接输出可操作的 affordance，减少绕物体一周的冗余扫描
- **抓取规划**：对遮挡的把手/握柄区域进行推理，直接指导 grasp 策略
- **人机交互**：开放词汇查询（"where to place a cup"）提供直观的任务描述界面

#### 领域影响
- **短期**：CVPR 2026 发表，建立 affordance + 重建联合建模的新 SOTA
- **中期**：可能推动 affordance grounding 从"判别式"向"生成式"的范式转变
- **长期**：Affordance 驱动的主动感知可能成为机器人标准范式

### 方法优势详解

1. **稀疏体素融合的 O(1) 复杂度**：无论多少视角，conditioning token 数量恒定（最多 4096），而 multi-view diffusion 方法 token 数 O(N) 增长
2. **5 步采样**：比 TRELLIS 的 25 步快 5×，关键性支撑主动视角选择的实时需求
3. **深度条件化解决方向歧义**：TRELLIS 仅 RGB 导致规范化输出方向与输入不一致，深度+相机 extrinsics 保证输出与真实世界坐标系对齐
4. **BCE+Dice 损失**：更适合二元 affordance heatmap 的监督，优于 flow matching 默认的 MSE

### 局限性分析

1. **物体中心假设**：当前仅处理单一物体，需要 SAM3D 等外部分割支持多物体场景——但这引入了额外的模块和失败模式
2. **初始 affordance 的依赖**：主动视角选择的质量取决于初始预测，错误的初始估计可能误导后续选择
3. **严重遮挡下的几何失效**：极端的自遮挡导致重建阶段就失败，无法被后续模块纠正
4. **AUC 指标略低**：未微调视觉/文本编码器，语义对齐不如 Espresso-3D 等判别式方法（72.0 vs 79.0）
5. **训练需要 8×A100 GPU**：限制了社区广泛复现

### 适用性与场景分析

#### 适用场景
- 桌面物体操作（目标物体可被分割）
- 部分观测下的 grasp/place 规划
- 功能区域主动探索

#### 不适用场景
- 大规模场景（室内/室外 navigat ion）
- 高度纠缠的多物体堆叠
- 实时闭环控制（7s 推理延迟）

## 与相关论文对比

### Espresso-3D / Affogato — 最直接对比

| 对比维度 | Espresso-3D | Affostruction |
|----------|-------------|---------------|
| 输入 | 完整点云 | 部分 RGBD |
| 方法 | 判别式（focal BCE） | 生成式（flow matching + BCE+Dice） |
| 未观测区域 | 无法处理 | 可推理 |
| 多模态建模 | 单模态 | 天然多模态 |
| 完整几何 aIoU | 13.6 | **19.1** |
| 部分观测 aIoU | 0.60 | **9.26** |

### TRELLIS — 基础模型

| 对比维度 | TRELLIS | Affostruction |
|----------|---------|---------------|
| 输入 | 单 RGB | 多 RGBD + 相机参数 |
| 输出 | 视觉外观（mesh/3DGS） | 几何 + affordance heatmap |
| 方向对齐 | 规范空间（不规范） | 真实世界坐标系 |
| 多视角利用 | 不支持 | O(1) 稀疏体素融合 |
| Toky4K IoU | 19.49 | **32.67** |

### MCC — 重建基线

| 对比维度 | MCC | Affostruction |
|----------|-----|---------------|
| 方法 | 判别式深度补全 | 生成式 flow matching |
| 输出 | 占用网格（无 mesh） | 稀疏体素 + mesh |
| 速度 | 35.37s（逐 chunk 处理） | **7.20s** |
| Toky4K IoU | 21.11 | **32.67** |

## 技术路线定位

### 所属技术路线
**3D Affordance Grounding + 生成式重建的交叉**

本质上是将两个独立发展的路线——(1) 开放词汇 3D affordance grounding (OpenAD → PointRefer → Espresso-3D) 和 (2) 稀疏体素 3D 生成 (TRELLIS → XCube → 3DTopia-XL)——首次深度融合。

### 技术路线发展历程
```
3D AffordanceNet (固定标签)
  → OpenAD (开放词汇, 判别式)
  → PointRefer (3D点引用, 判别式)
  → Espresso-3D / Affogato (规模化标注, 判别式)
  → Affostruction (生成式, 重建+affordance统一, 主动视角)
  → 未来: Affordance + 机器人操作闭环
```

### 本文在技术路线中的位置
- **承上**：继承 TRELLIS 的稀疏体素生成框架和 Affogato 的开放词汇 affordance 标注
- **启下**：为 "生成式几何 + 功能理解" 开辟路线，affordance 驱动的主动感知提供了新范式
- **关键节点**：从 "分别在完整形状上做判别式预测" → "从部分观测做生成式联合推理" 的关键转折

## 未来工作建议

### 作者建议的未来工作
1. **错误感知的视角选择**：引入 semi-supervised learning 的错误检测策略
2. **多物体联合重建**：与 SAM3D 的深度集成
3. **机器人操作应用**：将预测的 affordance 直接用于操作规划

### 基于分析的未来方向
1. **Text-conditioned 重建**：将文本查询反馈到 Stage 1，使重建本身关注与功能相关的区域
2. **实时推理**：模型蒸馏 + 进一步减少采样步数（当前 5 步→可能 1-2 步）
3. **Affordance-aware 的物体姿态估计**：利用 affordance 先验改进遮挡物体的 6D 姿态估计
4. **多模态 affordance 的显式建模**：当前 flow 隐含建模多模态，可考虑用 mixture model 显式输出多个 grasp/place 候选

## 综合评价

### 价值评分

#### 总体评分
**8.5/10** — 一项思路清晰、执行扎实的工作，将两个独立方向（生成式重建 + affordance）进行创造性融合，在关键指标上实现大幅超越。闭环的 affordance 驱动视角选择是一个漂亮的设计，部分观测下的 9.26 aIoU（2× 于 SOTA）是最具说服力的结果。

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 生成式重建+affordance 的统一思路新颖，flow-based 多模态建模有洞察；但每个单独组件（TRELLIS、flow matching、CLIP）均为现有技术 |
| 技术质量 | 9/10 | 稀疏体素融合设计精巧，随机多视角训练解决了一个关键的实际问题；BCE+Dice 损失选择合理 |
| 实验充分性 | 9/10 | 3 个层次的实验（重建、完整affordance、部分affordance）+ 视角选择 + 效率 + 消融 + 多物体 + 失败分析 |
| 写作质量 | 8/10 | 结构清晰，核心贡献表述准确；但部分细节（如 BCE+Dice 的权重）未在正文交代 |
| 实用性 | 8/10 | 7.2s 推理可在非实时场景使用；但 8×A100 训练 + object-centric 假设限制了即插即用性 |

### 值得关注的技术点
- **稀疏体素融合的 O(1) 复杂度**：相比 multi-view diffusion 的 O(N) 拼接，这是多视角推理的重大设计改进
- **5 步采样即收敛**：证明了 flow matching + 良好条件化可以在极少步骤中高质量生成
- **随机多视角训练策略**：简单但有效，模型从未见过 8 视角的情况下推理时最多可到 12 视角
- **Affordance 闭环**：预测→指导采集→改进预测的正反馈设计

> [!tip] 关键启示
> Affordance grounding 不应与几何重建解耦——两者在统一的稀疏体素空间中进行生成式建模，既能互补（几何约束 affordance 位置，affordance 引导视角），又能产生远超分离式方案的结果。

> [!warning] 注意事项
> - 方法的 object-centric 假设是当前最大限制——真实机器人场景中物体分割本身就是难题
> - 主动视角选择的 success 高度依赖初始 affordance 估计的质量，初始错误可能导致正反馈变成负反馈
> - 判别式方法在 AUC/SIM 等软指标上仍有优势（编码器微调），但 aIoU（空间精度）是操作任务更相关的指标

> [!success] 推荐指数
> ⭐⭐⭐⭐ 值得仔细阅读。这是 affordance grounding 方向的新范式工作，特别适合研究机器人操作中功能感知的同学。核心 insight（将生成式重建与功能理解统一）具有启发意义。

## 相关论文

### 直接相关
- [[Affogato / Espresso-3D]] - 开放词汇 affordance grounding 的 SOTA 方法和规模化标注
- [[TRELLIS]] - 稀疏体素 3D 生成基础模型，Affostruction 的基础架构
- [[OpenAD]] - 开放词汇 3D affordance 的早期工作
- [[PointRefer]] - 3D 点引用 affordance grounding

### 背景相关
- [[DINOv2]] - 视觉特征提取 backbone
- [[CLIP]] - 文本查询编码
- [[MCC]] - RGBD 多视角补全方法
- [[DUSt3R]] - 前馈多视角重建

### 后续工作
- [[SAM3D]] - 3D 分割模型，用于多物体扩展
- [[Flow Matching]] - 生成建模的理论基础
- [[Rectified Flow]] - TRELLIS 使用的生成框架

## 外部资源
- [Project Page](https://chrockey.github.io/Affostruction/)
- [arXiv](https://arxiv.org/abs/2601.09211)
