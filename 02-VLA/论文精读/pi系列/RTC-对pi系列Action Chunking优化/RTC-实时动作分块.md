---
date: "2026-05-12"
paper_id: "arXiv:2506.07339"
title: "RTC: Real-Time Execution of Action Chunking Flow Policies"
authors: "Kevin Black, Manuel Y. Galliker, Sergey Levine"
domain: "VLA_Embodied"
tags:
  - 论文笔记
  - VLA_Embodied
  - Real-Time-Inference
  - Action-Chunking
  - Flow-Matching
  - Inpainting
  - Inference-Time-Algorithm
  - π0-series
quality_score: "8.7/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# RTC: Real-Time Execution of Action Chunking Flow Policies

## 核心信息
- **论文ID**：arXiv:2506.07339
- **作者**：Kevin Black (PI + UC Berkeley), Manuel Y. Galliker (PI), Sergey Levine (PI + UC Berkeley)
- **机构**：Physical Intelligence & UC Berkeley
- **发布时间**：2025-06-09（v2: 2025-12-05）
- **会议/期刊**：NeurIPS 2025
- **链接**：[arXiv](https://arxiv.org/abs/2506.07339) | [PDF](https://arxiv.org/pdf/2506.07339) | [项目页面](https://pi.website/research/real_time_chunking)
- **引用**：--

## 摘要翻译

### 英文摘要
Modern AI systems, especially those interacting with the physical world, increasingly require real-time performance. However, the high latency of state-of-the-art generalist models, including recent VLAs, poses a significant challenge. While action chunking has enabled temporal consistency in high-frequency control tasks, it does not fully address the latency problem, leading to pauses or out-of-distribution jerky movements at chunk boundaries. This paper presents a novel inference-time algorithm that enables smooth asynchronous execution of action chunking policies. Our method, real-time chunking (RTC), is applicable to any diffusion- or flow-based VLA out of the box with no re-training. It generates the next action chunk while executing the current one, "freezing" actions guaranteed to execute and "inpainting" the rest.

### 中文翻译
现代AI系统，特别是与物理世界交互的系统，越来越需要实时性能。然而，最先进的通用模型（包括最近VLA模型）的高延迟构成了重大挑战。虽然动作分块（action chunking）使高频控制任务的时间一致性成为可能，但它并未完全解决延迟问题——导致chunk边界处的暂停或分布外（OOD）抖动。本文提出了一种新的推理时算法，使动作分块策略能够平滑异步执行。该方法（RTC）开箱即用于任何基于扩散或流的VLA，无需重新训练。它在执行当前chunk的同时生成下一个chunk——"冻结"确定会执行的动作，并"inpaint"其余动作。

### 核心要点提炼
- **研究背景**：现代VLA参数量巨大（数十亿），推理延迟远超控制周期（如π₀ KV cache prefill需46ms，OpenVLA最快要321ms），同步推理导致机器人执行时的暂停和分布偏移
- **研究动机**：动作分块（action chunking）虽保证了时间一致性，但chunk过渡处的不连续导致抖动和OOD动力学；简单的平滑（如temporal ensembling）可能产生无效动作
- **核心方法**：将异步chunk生成建模为inpainting问题——用pseudoinverse guidance（GDM）在流匹配去噪过程中引导新chunk与旧chunk的冻结前缀保持一致，引入soft masking实现跨chunk连续性
- **主要结果**：仿真12个高动态任务中显著超越所有基线（naive async、BID、Temporal Ensembling）；真实世界6个双臂操作任务中吞吐量最优，且对最高200ms注入延迟完全鲁棒
- **研究意义**：提供了首个实用的大模型实时控制推理框架——纯推理时算法，无需重新训练，直接适用于现有flow/diffusion-based VLA

## 研究背景与动机

### 领域现状
VLA模型通过动作分块（action chunking）实现了灵巧操作的最先进结果，但面临根本性的实时性挑战：

- **模型规模与延迟**：π₀（3B参数）在RTX 4090上KV cache prefill需46ms（50Hz控制周期仅20ms）；OpenVLA（7B）即使在A100上优化后仍需321ms
- **远程推理开销**：移动操作场景下，网络延迟轻易超过控制周期
- **同步推理的问题**：执行完chunk后暂停等待新chunk生成——导致可见的停顿、动力学分布偏移

### 现有方法的局限性

![[bifurcation.pdf|800]]

> 图1：chunk分叉示意。旧chunk（黑色）计划从障碍物上方通过，新chunk（红色）计划从下方通过。naive async直接跳转导致极高OOD加速度；Temporal Ensembling（插值）降低了加速度但产生无效动作（穿过障碍物）。

1. **Naive async**：在旧chunk执行中途启动推理，新chunk可用时直接切换——chunk间的任意不连续导致剧烈抖动
2. **Temporal Ensembling (TE)**：对多个chunk的重叠动作取平均——在多模态分布下平均动作可能是无效动作
3. **Bidirectional Decoding (BID)**：用rejection sampling保持连续性——计算量巨大（需采样64个chunk），且效果不如RTC
4. **模型加速方向**：减少推理步数/蒸馏等——无法突破"一次前向传播比控制周期长"的根本限制

### 研究动机
**实时系统必须在产生连续控制信号的同时，融入最新观测，不扰动环境的自然动力学。** 核心矛盾：大模型推理慢→需要异步执行→chunk间不连续→需要一种机制保证新旧chunk的一致性。

## 研究问题

### 核心研究问题
1. 如何在异步推理框架下保证新旧action chunk之间的连续性（无抖动）？
2. 如何在不重新训练的情况下，使flow/diffusion-based VLA实现实时控制？
3. RTC在不同推理延迟下的鲁棒性如何？
4. RTC在真实世界灵巧操作任务上的效果如何？

## 方法概述

### 核心思想
**将实时chunking建模为inpainting问题。** 新chunk生成时，旧chunk的前d步（inference delay期间确定被执行的动作）作为"冻结前缀"；flow matching的去噪过程中，通过pseudoinverse guidance（GDM）引导新chunk在与冻结前缀一致的前提下完成其余部分。Soft masking将guidance扩展为指数衰减的软权重，确保全跨chunk连续性。

### 方法框架

#### 整体架构

![[diagram.pdf|800]]

> 图2：RTC系统图解。关键概念：execution horizon (s)、inference delay (d)、frozen region（权重1）、intermediate region（指数衰减权重）、fresh generation region（权重0）。新chunk在背景线程中生成，同时前台继续执行旧chunk。

#### 系统流程

**Algorithm 1: Real-Time Chunking**

核心流程（`GETACTION` + `INFERENCELOOP` + `GUIDEDINFERENCE`）：

1. **前台** (`GETACTION`)：每个控制周期返回一个动作，更新共享观测
2. **后台** (`INFERENCELOOP`)：在背景线程中持续运行，s步后启动新chunk推理
3. **Guided Inference**：使用GDM gradient correction + soft masking

```
输入: flow policy π, H=预测horizon, smin=最小执行horizon
      
GETACTION(onext):          # 每个控制周期调用一次
  Acur[t] → 执行第t个动作
  更新共享观测ocur = onext

INFERENCELOOP:             # 后台循环
  wait until t ≥ smin
  Aprev = Acur[s:H-1]      # 移除已执行的s个动作
  d = max(past_delays)     # 保守估计下次延迟
  Anew = GUIDEDINFERENCE(π, o, Aprev, d, s)
  Acur = Anew              # 原子交换

GUIDEDINFERENCE(π, o, Aprev, d, s):
  计算soft mask W (Eq.5)
  初始化A0 ~ N(0, I)
  for τ = 0→1 (n步):
    fA1(A) = A + (1-τ)·v(A, o, τ)     # 去噪函数
    e = (Aprev - fA1(A)) · diag(W)     # 加权误差
    g = e · ∂fA1/∂A                     # VJP via autodiff
    A ← A + 1/n·[v(A,o,τ) + min(γ, (1-τ)/σ²)·g]  # 积分+guidance
  return A1
```

#### 核心技术1：Inpainting with Flow Matching

基于Pokle et al.的伪逆guidance（ΠGDM），推导出流匹配的gradient correction：

$$v_\text{GDM}(A^\tau, o_t, \tau) = v(A^\tau, o_t, \tau) + \min\left(\gamma, \frac{1-\tau}{\sigma_\tau^2}\right) \cdot \nabla_{A^\tau} \left\|Y - \hat{A}^1(A^\tau)\right\|^2_{\text{diag}(W)}$$

其中 $\hat{A}^1 = A^\tau + (1-\tau)v(A^\tau, o_t, \tau)$ 是最终去噪动作的估计，$Y$ 是目标（旧chunk的剩余动作），$W$ 是mask。Guidance weight clipping ($\gamma$) 是作者的改进——在小去噪步数下防止不稳定。

#### 核心技术2：Soft Masking

![[hard_vs_soft.pdf|800]]

> 图3：Hard vs Soft masking对比。hard masking仅用d个冻结动作，guidance信号弱，chunk间仍可能不连续；soft masking用指数衰减权重覆盖所有H-s个重叠动作，新chunk更平滑地衔接。

Soft mask定义：

$$W_i = \begin{cases} 1 & \text{if } i < d \\ c_i & \text{if } d \leq i < H-s \\ 0 & \text{if } i \geq H-s \end{cases}, \quad c_i = \frac{e^{\frac{H-s-i}{H-s-d}} - 1}{e - 1}$$

- 前d个动作权重为1（确定会执行，必须匹配）
- 中间区域指数衰减（越远的未来越不确定）
- 后s个动作权重为0（旧chunk无覆盖，需全新生成）

#### 关键技术约束

- 执行horizon满足：$d \leq s \leq H - d$（确保新chunk在需要前可用）
- 推理延迟缓冲区：保存最近b次延迟，保守取max
- 实际执行horizon：$s = \max(d, s_\text{min})$（随延迟自适应调整）

## 实验结果

### 实验目标
验证RTC在（1）高动态仿真任务、（2）不同推理延迟、（3）真实世界灵巧操作上的性能和速度

### 仿真基准（Kinetix）

![[plots_sim.pdf|800]]

> 图4：12个Kinetix高动态环境（car_launch, cartpole_thrust, catapult, catcher_v3, chain_lander, grasp_easy, h17_unicycle, hard_lunar_lander, mjc_half_cheetah, mjc_swimmer, mjc_walker, trampoline）。

**实验设置**：
- 用RPO训练专家策略→生成1M transitions数据集
- 训练action chunking flow policy（H=8，4层MLP-Mixer）
- 2048次rollout/data point，模拟延迟d=0到4

**基线**：
- Naive async：不关注旧chunk，直接切换
- BID：rejection sampling（batch=32, mode=3）
- TE：多chunk重叠动作平均

**关键结果**：
- RTC在所有延迟水平上表现最优，且随延迟增加与BID的差距拉大
- Soft masking优于hard masking（尤其在较小的d时）
- RTC是唯一能从更短execution horizon中持续获益的方法（证明其跨chunk连续性允许更快的闭环修正）
- TE在所有延迟下表现差——在多模态基准中，"平均动作"不一定是有效动作

### 真实世界实验

![[plots_real.pdf|800]]

> 图5：真实世界6个双臂操作任务结果。

**设置**：
- 基础模型：π₀.₅ VLA（H=50, Δt=20ms, n=5去噪步）
- 模型延迟：baseline 76ms, RTC 97ms（需额外VJP计算）
- LAN远程推理：额外10-20ms网络延迟
- 三种延迟条件：+0ms, +100ms, +200ms注入延迟（模拟更大模型/云推理）

**6个任务**：

| 任务 | 步数 | 时间限制 | 特点 |
|------|------|---------|------|
| Light Candle | 5 | 40s | 最高精度要求（划火柴），无retry |
| Plug Ethernet | 6 | 120s | 精细对准（插网线） |
| Make Bed (mobile) | 3 | 200s | 移动操作，最难任务 |
| Shirt Folding | 1 | 300s | 高精度灵巧操作 |
| Batch Folding | 4 | 300s | 多样衣物折叠 |
| Dishes in Sink (mobile) | 8 | 300s | 长序列移动操作 |

#### 总吞吐量

![[all_tasks_real.pdf|800]]

> 图6：总任务吞吐量（tasks/min）—速度与性能的综合度量。RTC在所有延迟条件下最优，+100ms和+200ms时优势具有统计显著性。RTC对延迟完全鲁棒（无性能退化），Synchronous随延迟线性退化，两种TE变体在+100ms和+200ms时因剧烈振荡触发机器人保护性停止。

#### 各任务详细分析

![[candle_frame1.jpg|200]] ![[candle_frame2.jpg|200]] ![[candle_frame3.jpg|200]] ![[candle_frame4.jpg|200]]

> 图7：Light Candle任务帧序列——最高精度的任务，也是唯一不允许retry的任务。RTC在此任务上最终得分大幅领先，反映了更高的整体成功率。

关键发现：
- **即使去除推理暂停时间**，RTC也比Synchronous更快完成任务——说明RTC不仅提速，还减少错误和retry
- **Light Candle**：精度最敏感的任务，RTC优势最大（无retry容忍）
- **Make Bed**：最困难的任务（操作枕头尤其困难），RTC同样有显著效果
- **两者TE变体**在+100ms/+200ms延迟下因振荡过大完全无法运行

### 消融实验

![[beta_sim.pdf|800]]

> 图8：Guidance weight clipping和soft masking消融。γ clipping防止小去噪步数下的不稳定；soft masking在低延迟和短execution horizon下提供额外性能增益。

![[plots_sim_schedule_sweep.pdf|800]]

> 图9：Soft masking衰减schedule sweep。验证了指数衰减schedule的有效性。

## 深度分析

### 研究价值评估

#### 理论贡献
- **贡献1：将实时动作生成建模为inpainting问题**
  - 创新点：首次将扩散/流模型的inpainting技术应用于实时控制场景
  - 学术价值：连接了生成模型inpainting文献与机器人控制文献
  - 影响范围：所有使用diffusion/flow-based策略的实时控制系统

- **贡献2：Soft masking机制**
  - 创新点：用指数衰减的软权重实现全跨chunk连续性，超越了简单的"冻结前缀+inpaint"方案
  - 学术价值：证明了inpainting中mask设计对控制质量的关键影响

- **贡献3：新基准**
  - 创新点：12个Kinetix高动态环境——传统的quasi-static基准已被饱和
  - 学术价值：为实时控制算法的评估提供了标准化的动态测试平台

#### 实际应用价值
- **应用场景1：大模型远程推理的实时控制**
  - 适用性：所有需要将大VLA部署到移动机器人上，通过网络远程推理的场景
  - 优势：对200ms+延迟完全鲁棒，使"云端大模型+边缘机器人"架构可行
  - 潜在影响：解耦模型规模和部署位置——最大模型可运行在数据中心，机器人端仅需执行

- **应用场景2：需要高精度时序协调的任务**
  - 适用性：划火柴、精细装配等对动作连续性要求极高的任务
  - 优势：相比同步推理，不仅更快而且更稳（减少chunk边界处的动力学扰动）

#### 领域影响
- **短期影响**：为现有flow/diffusion-based VLA提供即插即用的实时推理方案
- **中期影响**：可能成为VLA部署的标准推理模式
- **长期影响**：推理时算法（inference-time algorithm）作为弥补训练与部署差距的通用范式

### 方法优势详解

#### 优势1：零训练成本（plug-and-play）
- **描述**：纯推理时算法，不需要重新训练、微调或修改模型架构
- **技术基础**：利用预训练flow matching模型的原生inpainting能力
- **适用性**：适用于任何flow/diffusion-based VLA

#### 优势2：延迟鲁棒性
- **描述**：对最高200ms+注入延迟完全鲁棒，性能不退化
- **技术基础**：保守延迟估计 + soft masking中的指数衰减天然容忍不确定性
- **对比**：Synchronous随延迟线性退化，TE在高延迟下完全失败

#### 优势3：速度与精度的双重提升
- **描述**：不仅执行更快（+20%吞吐量），任务完成质量也更高
- **技术基础**：异步执行消除暂停→更快的任务完成；跨chunk连续性→更少的错误和retry
- **实验验证**：去除推理暂停时间后RTC仍比Synchronous更快完成任务

### 局限性分析

#### 局限1：仅适用于diffusion/flow-based策略
- **描述**：RTC依赖迭代去噪过程的inpainting能力
- **原因**：自回归模型不经过噪声空间，无法应用GDM guidance
- **影响**：FAST-based自回归VLA（如π₀-FAST）无法使用RTC
- **可能方案**：探索将自回归采样与噪声空间连接的方案

#### 局限2：计算开销
- **描述**：VJP（vector-Jacobian product）计算增加了每次去噪步骤的开销
- **数据**：RTC的模型延迟为97ms vs baseline 76ms（增加约28%），但wall-clock执行时间减少20%
- **权衡**：净效果仍为正——时间节省远大于单步开销增加

#### 局限3：真实世界实验限于特定设置
- **描述**：仅在π₀.₅ + 双臂操作 + 位置控制上验证
- **原因**：Physical Intelligence的专有模型和硬件
- **影响**：在力控或腿式运动等更动态场景的通用性需进一步验证

#### 局限4：Soft masking超参数
- **描述**：指数衰减schedule的特定形式可能需要针对不同任务调整
- **原因**：最优的衰减速率可能依赖任务动态特性
- **影响**：虽然论文展示了鲁棒性，但在极端不同的任务分布下可能需要重新调整

### 适用性与场景分析

#### 适用场景
- **远程推理的大模型部署**：云端VLA + 边缘机器人执行
- **高精度时序敏感任务**：划火柴、精细装配、动态操作
- **需要闭环修正的动态环境**：力控任务、移动操作
- **现有flow/diffusion VLA的快速部署**：无需任何重新训练

#### 不适用场景
- **自回归VLA**（如π₀-FAST）：不是diffusion/flow-based
- **低延迟本地推理已满足实时性**：额外计算开销不划算
- **极简单的quasi-static任务**：同步推理可能已足够

## 与相关论文对比

### [[pi0_Vision-Language-Action_Flow_Model|π₀]] - Flow Model for Robot Control

#### 方法对比
| 对比维度 | π₀（同步推理） | RTC + π₀ |
|----------|---------------|----------|
| 推理方式 | 执行完chunk→等待→新chunk | 执行中后台生成新chunk→inpainting切换 |
| 延迟鲁棒性 | 线性退化 | 完全鲁棒（200ms+） |
| 吞吐量 | Baseline | +20% |
| 部署成本 | 无额外 | 推理时算法，零训练 |

**关系分析**：RTC是π₀/π₀.₅的推理时增强——不改模型，直接提升实时性能。

### [[pi0.5_Open_World_Generalization|π₀.₅]] - Open-World Generalization

**关系分析**：RTC使用π₀.₅作为真实世界实验的基础模型。两者是正交贡献：π₀.₅解决"模型能做什么"（泛化），RTC解决"模型做得快且稳"（实时执行）。

### [[FAST_Efficient_Action_Tokenization_for_VLA|FAST]] - Action Tokenization

#### 互补关系
| 对比维度 | FAST-based VLA | RTC |
|----------|---------------|-----|
| 核心思路 | 减少token数→加速推理 | 异步+inpainting→隐藏延迟 |
| 加速方式 | 训练时（tokenizer设计） | 推理时（inference algorithm） |
| 适用范围 | 自回归VLA | Diffusion/flow VLA |
| 组合可能性 | 不直接兼容（不同范式） | -- |

### Temporal Ensembling (Zhao et al., 2023)

**关系分析**：RTC的论文中TE是主要基线之一。两者都试图解决chunk间的连续性，但方法论根本不同：
- TE：对多chunk的重叠动作取平均（信号处理思路）
- RTC：通过inpainting约束新chunk与旧chunk一致（生成模型思路）

在多模态动作分布下，TE的理论弱点被暴露——"平均动作"不一定是"有效动作"。

## 技术路线定位

### 所属技术路线
本文属于**实时机器人推理**技术路线，聚焦于用推理时算法弥补大模型延迟与实时控制需求之间的gap。

### 技术路线发展历程
```
PID控制（经典）→ MPC（优化+rolling horizon）→ Action Chunking（学习+chunk）→ RTC（学习+chunk+异步inpainting）
         ↓                     ↓                              ↓                           ↓
    固定频率              warm-start+并行化              同步执行+暂停          完全异步+跨chunk连续性
```

### 本文在技术路线中的位置
- **承上**：继承action chunking范式和flow matching策略
- **启下**：为更大规模模型（千亿参数级）的实时部署铺路；推理时算法+模型训练解耦的新范式
- **关键节点**：证明了"不需要缩小模型也能实现实时控制"——通过推理时算法而非模型压缩来解决延迟问题

## 未来工作建议

### 作者建议的未来工作
1. 探索RTC与分级架构（System 1/System 2）的结合
2. 将RTC扩展到腿式运动等更动态的场景
3. 研究级联控制理论与学习式action chunking策略的交汇点

### 基于分析的未来方向
1. **RTC + 自回归VLA**：将inpainting思想扩展到离散token空间（如利用masked token prediction），使FAST-based VLA也能受益
2. **自适应soft masking schedule**：学习性地预测最优衰减速率——不同任务可能需要不同的连续性程度
3. **RTC + KI联合优化**：在KI训练中显式考虑RTC推理模式（如训练时模拟异步执行），进一步提升协同效果
4. **RTC + 人类数据**：结合π₀.₅+ego，使从人类视频学到的技能也能实时执行

## 我的综合评价

### 价值评分

#### 总体评分
**8.7/10** - 优雅的推理时解决方案，具有坚实的理论基础（inpainting+guidance），实验全面（12仿真+6真机），实用价值高。主要扣分在于适用范围限于flow/diffusion模型，以及计算开销。

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | 首次将inpainting范式应用于实时控制，soft masking机制精巧，推理时算法的定位新颖 |
| 技术质量 | 9/10 | 数学推导清晰，算法伪代码完整，guidance weight clipping是必要且有效的改进 |
| 实验充分性 | 9/10 | 480次真机测试（28小时robot time），12仿真环境×2048 trials，延迟鲁棒性sweep |
| 写作质量 | 8/10 | 清晰但略长（引入部分的AI系统例子略显冗余），图表质量高 |
| 实用性 | 9/10 | 即插即用、零训练、对现有VLA直接适用——实用价值极高 |

### 重点关注

#### 值得关注的技术点
- GDM pseudoinverse guidance中的guidance weight clipping（小去噪步数下防不稳定）
- Soft masking的指数衰减schedule（$W_i$公式）
- 执行horizon的自适应调整（$s = \max(d, s_\text{min})$）
- 延迟缓冲区的保守估计策略（取max而非mean）

#### 需要深入理解的部分
- VJP（vector-Jacobian product）的计算开销 vs 性能增益的tradeoff
- Soft masking衰减schedule在不同任务动态下的最优选择
- 与MPC的warm-start策略的理论联系（都是利用先前规划加速当前规划）

## 相关论文

### 直接相关
- [[pi0.5_Open_World_Generalization|π₀.₅]] - 真实世界实验的基础模型
- [[pi0_Vision-Language-Action_Flow_Model|π₀]] - flow matching action expert架构
- Diffusion Policy (Chi et al., 2023) - 动作分块的扩散策略基础

### 背景相关
- BID (Liu et al., 2025) - 最相关的前序工作（rejection sampling based）
- ΠGDM (Pokle et al., 2024) - inpainting的pseudoinverse guidance基础
- Temporal Ensembling (Zhao et al., 2023) - 主要基线
- Consistency Policy - 蒸馏加速方向
- Streaming Diffusion Policy - 少步数去噪方向

### 后续工作
- 尚无公开的后续论文

## 外部资源
- [项目页面与视频](https://pi.website/research/real_time_chunking) - 含真实世界任务的完整视频
- [Kinetix Simulator](https://github.com/google-research/kinetix) - 使用的仿真环境
- Supplementary material含仿真基准的完整代码

## 关联笔记

- [[π₀]] — RTC 直接适用的 Flow Matching VLA
- [[π₀-FAST]] — FAST 自回归与 RTC 的兼容性对比
- [[π-0.6-RECAP]] — π*₀.₆ 中 RTC 的实际应用
- [[扩散策略]] — 扩散/Flow 动作生成的基础（RTC 的投影操作对象）
- [[VLA模型总览]] — 不同动作生成方式的实时性对比

> [!tip] 关键启示
> **大模型的实时性不需要通过缩小模型来实现——推理时算法可以优雅地解决延迟问题。** RTC的核心洞察是：将异步chunk生成建模为inpainting——利用flow matching的原生inpainting能力（通过pseudoinverse guidance），在不重新训练的情况下实现新旧chunk的平滑过渡。这为"大模型远程推理+边缘执行"的部署架构提供了关键的技术支撑。

> [!warning] 注意事项
> - RTC仅适用于diffusion/flow-based VLA，不适用于自回归VLA
> - VJP计算增加了每次去噪步骤的约28%开销（但净wall-clock时间仍减少20%）
> - Soft masking的指数衰减schedule可能需要针对不同任务动态调整
> - 延迟估计保守取max——如果延迟波动很大，可能导致过于保守的执行horizon
> - 真实世界实验仅验证了位置控制设置，力控/腿式运动需进一步验证

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！本文是"推理时算法"解决部署问题的典范——不需要重新训练，直接让现有VLA实现实时性能。对于部署大模型到实时控制系统的实践者，RTC提供了一个理论基础扎实、工程上实用的完整方案。论文中的inpainting视角和soft masking设计也值得生成模型研究者关注。
