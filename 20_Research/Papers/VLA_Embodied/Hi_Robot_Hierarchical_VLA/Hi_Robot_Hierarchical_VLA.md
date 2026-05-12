---
date: "2026-05-12"
paper_id: "arXiv:2502.19417"
title: "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models"
authors: "Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, Chelsea Finn"
domain: "VLA_Embodied"
tags:
  - 论文笔记
  - VLA_Embodied
  - VLA
  - Hierarchical-Policy
  - System1-System2
  - Instruction-Following
  - Synthetic-Data
  - Human-Robot-Interaction
  - π0-series
quality_score: "9.0/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models

## 核心信息
- **论文ID**：arXiv:2502.19417
- **作者**：Lucy Xiaoyang Shi (PI + Stanford), Brian Ichter (PI), Michael Equi (PI), Liyiming Ke (PI), Karl Pertsch (PI + Berkeley), Quan Vuong (PI), James Tanner (PI), Anna Walling (PI), Haohuan Wang (PI), Niccolo Fusai (PI), Adrian Li-Bell (PI), Danny Driess (PI), Lachy Groom (PI), Sergey Levine (PI + Berkeley), Chelsea Finn (PI + Stanford)
- **机构**：Physical Intelligence, Stanford University, UC Berkeley
- **发布时间**：2025-02-26（v2: 2025-07-15）
- **会议/期刊**：ICML 2025
- **链接**：[arXiv](https://arxiv.org/abs/2502.19417) | [PDF](https://arxiv.org/pdf/2502.19417) | [项目页面](https://pi.website/research/hirobot)
- **引用**：--

## 摘要翻译

### 英文摘要
Generalist robots that can perform a range of different tasks in open-world settings must be able to not only reason about the steps needed to accomplish their goals, but also process complex instructions, prompts, and even feedback during task execution. In this work, we describe a system that uses vision-language models in a hierarchical structure, first reasoning over complex prompts and user feedback to deduce the most appropriate next step to fulfill the task, and then performing that step with low-level actions. In contrast to direct instruction following methods that can fulfill simple commands, our system can reason through complex prompts and incorporate situated feedback during task execution. We evaluate across three robotic platforms, demonstrating its ability to handle tasks such as cleaning messy tables, making sandwiches, and grocery shopping.

### 中文翻译
能够在开放世界环境中执行多种任务的通用机器人，必须不仅能推理完成目标所需的步骤，还要能处理复杂的指令、提示，甚至任务执行中的反馈。本文描述了一个以层级结构使用VLM的系统：首先对复杂提示和用户反馈进行推理，推断出最合适的下一步，然后用低层动作执行该步骤。与只能完成简单命令的直接指令跟随方法相比，本系统能够推理复杂提示并在任务执行中融入情境化反馈。我们在三个机器人平台上评估，展示了处理清理桌面、制作三明治、杂货购物等任务的能力。

### 核心要点提炼
- **研究背景**：现有VLA能跟随简单原子指令（"拿起可乐罐"），但无法处理开放式的复杂提示和实时用户反馈
- **研究动机**：机器人需要像人类一样灵活——理解复杂多步指令、接受中途修正、在新任务组合中泛化
- **核心方法**：层级VLM架构——高层VLM（"System 2"）推理复杂提示输出简单子任务命令 → 低层VLA（"System 1"）执行该命令输出动作；合成数据生成：用大VLM反向生成与机器人观测和技能标签匹配的假设用户提示
- **主要结果**：在3个机器人平台×3个任务领域上，指令准确率比GPT-4o高40%+，接近人类专家的高层指导水平
- **研究意义**：证明了层级VLM架构是实现开放领域指令跟随的有效路径——将语义推理与物理执行分离，各司其职

## 研究背景与动机

### 领域现状
VLA模型已展现出令人印象深刻的泛化和指令跟随能力，但其处理的语言命令通常是**简单、原子化的**（如"拿起杯子放在盘子上"）。真实世界的需求远不止于此——用户可能给出多步复杂指令、中途修改需求、或提供实时纠正反馈。现有方法要么无法处理复杂语言交互（flat VLA），要么物理灵巧性不足（基于API的VLM+预定义技能）。

### 现有方法的局限性

![[baselines_v2.pdf|800]]

> 图1：与GPT-4o和flat VLA的对比。Hi Robot在三个领域上的指令准确率平均高40%以上。

1. **Flat VLA（端到端）**：直接训练VLA跟随指令——适合原子指令，但无法解析复杂多步提示、中途修正和上下文约束
2. **API-based VLM（如GPT-4o）+ 预定义技能**：高层推理能力强但缺乏物理接地——频繁误识别物体、跳过子任务、忽略用户意图。GPT-4o一旦物理交互开始就失去上下文
3. **YAY Robot等交互系统**：限于单一人写提示和预定义修正类型——无法泛化到开放式的多样化语言交互

### 研究动机
类比Kahneman的"System 1 / System 2"认知模型：
- **System 1**（自动/快速）：执行简单命令、触发预学技能
- **System 2**（慎思/缓慢）：解析复杂任务、解释反馈、决定行动方案

机器人需要**两套系统同时工作**——高层推理决定"做什么"，低层执行决定"怎么做"。

## 研究问题

### 核心研究问题
1. 如何让机器人处理开放式的复杂语言指令（不只是原子指令）？
2. 如何融入实时用户反馈和修正（"那不是垃圾"、"leave it alone"）？
3. 如何获得足够的复杂交互训练数据（不可能靠人工标注所有可能的prompt）？
4. 层级架构是否优于flat VLA？

## 方法概述

### 核心思想
**用两个VLM构建层级推理系统**——高层VLM负责"thinking"（理解复杂prompt → 输出简单子任务命令），低层VLA负责"doing"（接收子任务命令 → 输出动作）。两者通过**语言接口**连接（高层"说"原子指令，低层执行）。关键创新：用合成数据生成方案解决高层模型的训练数据问题。

### 方法框架

#### 整体架构

![[block_diagram_v2.pdf|800]]

> 图2：Hi Robot层级架构。高层policy（VLM）处理开放指令和图像→生成低层语言命令cmd_t；低层policy（VLA，即π₀）用cmd_t、图像和机器人状态→输出动作块A_t。两者以不同频率运行（高层1Hz或事件触发，低层50Hz）。

#### 层级推理流程

**高层Policy** $\pi_H(\text{cmd}_t | I_t^1,...,I_t^n, \ell_t)$：
- 输入：多视角图像 + 开放提示/用户反馈 $\ell_t$
- 输出：原子命令 $\text{cmd}_t$（如"pick up the lettuce"）+ 可选语音回复 $u_t$
- 推理频率：每1秒或收到用户输入时触发
- 模型：PaliGemma-3B VLM fine-tuned

**低层Policy** $\pi_L(A_t | I_t^1,...,I_t^n, \text{cmd}_t, q_t)$：
- 输入：多视角图像 + 高层命令 $\text{cmd}_t$ + 机器人状态 $q_t$
- 输出：动作块 $A_t$（通过flow matching，与π₀相同）
- 推理频率：50Hz（标准VLA推理）
- 模型：PaliGemma-3B fine-tuned → π₀ VLA（带action expert）

**用户交互机制**：
- 用户可在任意时刻发文本/语音输入
- 触发高层推理立即重新计算cmd_t
- 高层可生成语音回复 $u_t$ → TTS播放给用户
- 交互完成后用户可示意恢复原任务

#### 数据采集与生成

![[data_gen_v3.pdf|800]]

> 图3：高层policy训练数据生成流程。从遥操作演示→分割为短技能（1-3秒）→用大VLM反向生成假设的复杂用户prompt和机器人回复。

**步骤1：人工标注** $\mathcal{D}_{labeled}$
- 遥操作采集完整演示episode
- 分割为1-3秒的短技能 $\text{cmd}_t$（如"pick up one piece of lettuce"、"move right arm to the left"）
- 记录对应的图像观测

**步骤2：合成数据生成** $\mathcal{D}_{syn}$
- 用大VLM（$\pi_{gen}$）输入：图像 + 技能标签 $\text{cmd}_t$
- 反向生成：假设的用户prompt $\ell_t$（如"Can you add some lettuce for me?"）+ 机器人语音回复 $u_t$
- 效果：生成多样化、组合式的语言提示（同一技能对应多种表达）

**步骤3：联合训练**
- 高层 $\pi_H$：在 $\mathcal{D}_{syn} \cup \mathcal{D}_{labeled}$ 上用交叉熵loss训练next-token prediction
- 低层 $\pi_L$：在 $\mathcal{D}_{labeled} \cup \mathcal{D}_{demo}$ 上用flow-matching loss（遵循π₀的训练方案）

### 关键设计决策

| 设计维度 | 选择 | 理由 |
|---------|------|------|
| 高层/低层模型 | 相同backbone（PaliGemma-3B） | 架构统一，但角色分离 |
| 层间接口 | 自然语言（$\text{cmd}_t$） | 可解释、可debug、灵活 |
| 高层频率 | 1Hz或事件触发 | 高层推理慢但不需高频 |
| 低层频率 | 50Hz（正常VLA频率） | 需要实时精确控制 |
| 合成数据 | 反向生成用户prompt | 解决人工标注多样性的瓶颈 |

## 实验结果

### 实验目标
验证Hi Robot在（1）复杂多步指令跟随、（2）实时用户反馈融入、（3）跨机器人平台和任务领域的泛化能力

### 任务与平台

![[tasks_v2.pdf|800]]

> 图4：三个任务领域。Table Bussing（单臂）、Sandwich Making（双臂）、Grocery Shopping（移动双臂）。

| 任务 | 机器人平台 | 物理挑战 | 语言挑战 |
|------|-----------|---------|---------|
| Table Bussing | 单臂 | 精细抓取（盘子边缘）、物体分离、用盘子倒垃圾 | "只清理垃圾不碰餐具"、"把所有黄色的东西收走"、"这不是垃圾" |
| Sandwich Making | 双臂 | 柔软/易碎食材操作（生菜、奶酪） | "做一个素食三明治"、"我对泡菜过敏"、"够了不用再加了" |
| Grocery Shopping | 移动双臂 | 移动操作、多物体抓取和放置 | "帮我去点薯片"、"去点甜的东西"、"我也想要KitKat" |

### 基线方法

| 方法 | 描述 |
|------|------|
| **Expert Human HL** | Oracle基线：人类手动输入高层命令（测试低层policy能力上限） |
| **GPT-4o HL** | 用GPT-4o API替代高层VLM（类比SayCan进阶版），prompt engineering列出最常用技能标签 |
| **Flat VLA** | 直接使用π₀低层policy，无高层、无合成数据 |
| **Flat VLA + synthetic data** | 低层policy直接用合成数据训练（处理复杂prompt），但无层级结构 |
| **Hi Robot w/o synthetic data** | 有层级结构但只训练在人工标注数据上 |

### 评估指标

- **Instruction Accuracy (IA)**：高层预测指令与人类意图的一致性（blind human evaluator评分）
- **Task Progress (TP)**：正确放置/配置的物体比例

每种方法×每个任务20次试验，总计大规模人工评估。

### 主要结果

![[qualitative_comparison_v2.pdf|800]]

> 图5：定性对比。GPT-4o常见失败模式：(a)误识别物体、(b)跳步子任务、(c)忽略用户意图。Hi Robot始终保持与机器人ongoing action和用户请求的一致性。

**核心发现1：Hi Robot在开放指令跟随上显著优于所有基线**
- 指令准确率平均比GPT-4o高40%+
- GPT-4o频繁在物理交互开始后失去上下文——发出无意义命令（"pick up bermuda triangle"）、将一切标记为"plate"/"spoon"
- Flat VLA完全不响应实时反馈

**核心发现2：Hi Robot展现出强情境化推理和反馈适应**
- 用户中途修正（"leave the rest"、"I also want a KitKat"）→ Hi Robot正确更新低层命令
- GPT-4o内部状态不一致——如在夹爪仍持物时命令抓取新物体，或过早切换任务
- Flat baseline无实时反馈响应能力

**核心发现3：跨平台和任务类型的泛化**
- 在单臂、双臂、移动双臂三个平台上均有效
- 处理多样物体（从易碎的奶酪片到高瓶子）同时遵守动态约束
- Flat baseline和GPT-4o在prompt变化时倾向回退到默认行为（收走所有物品、加入几乎所有配料）

**核心发现4：人类专家高层指导揭示低层policy足够强，瓶颈在高层推理**
- 用人类手动输入高层命令时低层policy几乎完美执行
- 这说明性能差异主要来自**推理能力**而非**执行能力**
- Hi Robot有效弥合了这一鸿沟——VLM高层+物理接地

### 消融实验

![[ablation_data_v2.pdf|800]]

> 图6：合成数据消融。无合成数据的模型忽略用户澄清（"this is not trash"）或包含禁止物品（pickles）；Hi Robot因合成数据中丰富的组合式语言覆盖而平滑适应。

**消融A：合成数据至关重要**
- 仅在人工标注数据上训练的高层policy → 忽略澄清、包含禁止物品
- 合成数据提供的**组合式语言多样性**是泛化的关键

![[ablation_hierarchy_v2.pdf|800]]

> 图7：层级结构 vs flat policy。层级方法有效融入用户反馈和部分指令；flat模型即使训练在相同合成数据上也倾向回退到默认行为。

**消融B：层级结构优于flat policy**
- Flat VLA + synthetic data虽然能处理复杂prompt，但中途修正时表现差
- 层级结构允许在每个高层步骤重新检查prompt——对多步连贯性和用户动态输入至关重要
- 分离高层推理与低层控制是beneficial的

## 深度分析

### 研究价值评估

#### 理论贡献
- **贡献1：将System 1/System 2认知模型实例化到VLA**
  - 创新点：两个VLM（几乎相同架构）+ 语言接口实现认知分层
  - 学术价值：证明了"推理/执行分离"可以用同质化的VLM实现，而非需要不同类型的模型
  - 影响范围：所有面向复杂交互的机器人系统设计

- **贡献2：合成数据反向生成方案**
  - 创新点：给定观测+技能标签 → 反向生成假设的用户prompt（而非正向：给定prompt → 生成技能）
  - 学术价值：解决了"如何低成本获取多样化交互训练数据"的关键瓶颈
  - 影响范围：所有需要大规模语言交互训练数据的系统

- **贡献3：系统级的全面评估**
  - 创新点：复杂多步prompt + 实时修正 + 跨平台 + blind human evaluation
  - 学术价值：为开放领域指令跟随提供了全面的评估基准和方法论

#### 实际应用价值
- **应用场景1：家庭服务机器人**
  - 适用性：用户需要灵活指定任务（"做一个素食三明治，不要番茄"）并实时修正
  - 优势：不需要用户学习如何给机器人下命令——自然对话即可

- **应用场景2：工业协作机器人**
  - 适用性：工人需要实时指导机器人调整行为
  - 优势：语言接口直观，减少编程需求

#### 领域影响
- **短期影响**：提供了层级VLA的可复现实现方案
- **中期影响**：语言接口的模块化设计 → 高层和低层可独立升级（如更强的VLM替换高层，更强的VLA替换低层）
- **长期影响**：将"推理"和"执行"解耦可能成为VLA设计的标准范式

### 方法优势详解

#### 优势1：模块化与可升级性
- **描述**：高层和低层通过语言接口连接，彼此独立
- **优势**：可以单独升级任一层——更强的VLM做高层推理（Gemma 3、GPT-5），或更强的VLA做低层执行
- **对比**：Flat VLA的端到端结构升级需要全盘重训

#### 优势2：可解释性
- **描述**：高层输出是可读的自然语言命令
- **优势**：debug时可直接看到机器人"在想什么"——哪个高层命令导致了哪个低层动作
- **对比**：Flat VLA是黑盒——无法理解为何做出特定动作

#### 优势3：数据效率
- **描述**：合成数据生成让同样的机器人演示覆盖多样化的语言表达
- **优势**：不需要为每一种prompt重新采集机器人数据

### 局限性分析

#### 局限1：高层和低层互相不知对方能力
- **描述**：两层的训练是分离的——高层不知道低层擅长或不擅长什么
- **原因**：训练时没有跨层反馈
- **影响**：高层可能发出低层无法执行的命令
- **可能方案**：训练时融入低层成功/失败的反馈信号

#### 局限2：高层推理频率固定
- **描述**：当前简单策略（每1秒或事件触发）
- **原因**：为简单性和可复现性
- **可能方案**：自适应推理——检测到子任务完成时立即触发，而非等待1秒

#### 局限3：合成数据的prompt engineering
- **描述**：生成多样化的合成prompt仍需要一定的prompt engineering
- **原因**：需要引导合成VLM生成符合实际情况的用户prompt
- **影响**：不同任务可能需要调整合成prompt模板

#### 局限4：评估依赖人工评分
- **描述**：IA和TP需要blind human evaluator
- **原因**：指令准确性的自动化评估困难
- **影响**：大规模迭代优化成本高、速度慢

### 适用性与场景分析

#### 适用场景
- **需要复杂语言交互的任务**：多步指令、用户约束、实时修正
- **需要新任务组合的泛化**：训练时未见过的提示组合
- **需要可解释性的部署**：需要理解机器人决策过程

#### 不适用场景
- **简单原子指令**：直接用flat VLA更简洁高效
- **对延迟极度敏感的任务**：高层推理引入额外延迟（虽然已异步运行）
- **不需要语言交互的纯自主任务**：层级结构的复杂性不必要

## 与相关论文对比

### [[pi0_Vision-Language-Action_Flow_Model|π₀]] - Flow Model for Robot Control

#### 方法对比
| 对比维度 | π₀（Flat VLA） | Hi Robot |
|----------|---------------|----------|
| 架构 | 单一VLA（无分层） | 两层VLM（高层+低层$\approx$π₀） |
| 指令复杂度 | 原子指令 | 开放复杂指令+实时反馈 |
| 推理机制 | 端到端 | System 2→System 1 |
| 语言交互 | 无 | 双向语音对话 |

**关系分析**：Hi Robot的低层直接使用π₀ VLA。核心贡献是在π₀之上添加了高层推理层。

### SayCan / PaLM-E

#### 方法对比
| 对比维度 | SayCan/PaLM-E | Hi Robot |
|----------|--------------|----------|
| 高层 | LLM（SayCan）/ VLM（PaLM-E） | Fine-tuned VLM |
| 低层 | 预定义技能（RL-trained） | 学习式VLA（端到端训练） |
| 物理灵巧性 | 有限（预定义技能受限于训练范围） | 高（VLA可产生复杂连续动作） |
| 语言接地 | 价值函数 + affordance | 图像观测直接输入VLM |

**关系分析**：Hi Robot相比SayCan，关键区别在于低层也是学习式的VLA→物理灵巧性大幅提升；相比PaLM-E，高层是fine-tuned VLM而非API-based，具有更好的物理接地。

### YAY Robot (Shi et al., 2024)

#### 方法对比
| 对比维度 | YAY Robot | Hi Robot |
|----------|-----------|----------|
| 高层模型 | 基于规则的命令映射 | Fine-tuned VLM |
| 提示类型 | 单一预定义prompt | 开放灵活prompt |
| 修正类型 | 人写固定修正集 | 开放合成修正 |
| 数据 | 仅人工标注 | 人工标注+合成数据 |

**关系分析**：Hi Robot可视为YAY Robot的全面升级——用VLM + 合成数据突破YAY Robot在语言多样性和开放性上的根本限制。

### [[pi0.5_Open_World_Generalization|π₀.₅]] - Open-World Generalization

**关系分析**：π₀.₅和Hi Robot重叠但焦点不同。π₀.₅专注于通过5类数据源协同训练实现物理泛化（新场景/新物体），Hi Robot专注于通过层级架构实现语言交互泛化。两者互补——未来可结合（π₀.₅的泛化能力 + Hi Robot的层级交互能力）。

### [[RTC_Real_Time_Action_Chunking|RTC]] - Real-Time Chunking

**关系分析**：RTC在论文Discussion中提及——层级架构（高层慢/低层快）天然适合与RTC结合。更高层推理可运行在更低的频率，低层用RTC实现异步实时执行。两者是正交的、可叠加的技术。

### [[pi0.6_RECAP_VLA_Learns_From_Experience|π₀.₆ RECAP]] - VLA Learns From Experience

**关系分析**：π₀.₆的自改进能力与Hi Robot的层级架构结合——高层在推理时可以利用低层的RECAP改进经验做出更好的子任务选择。

## 技术路线定位

### 所属技术路线
本文属于**层级VLA**技术路线的里程碑工作，连接了"语言模型推理"和"VLA物理执行"两个领域。

### 技术路线发展历程
```
LLM Planner（SayCan等）→ VLM Planner（PaLM-E等）→ 层级VLA（Hi Robot）
     ↓                        ↓                        ↓
  LLM+预定义技能         VLM+预定义技能          VLM+VLA（全可学习）
  物理灵巧性差            物理灵巧性差             强物理灵巧性+灵活语言
```

### 本文在技术路线中的位置
- **承上**：继承π₀的VLA架构、YAY Robot的语言交互思路、SayCan的层级思想
- **启下**：为"全VLM"架构铺路——未来可合并高层和低层到同一模型，仅在推理时区分System 1/System 2
- **关键节点**：证明了层级VLA的可行性和优势——是两个领域融合的拐点

## 未来工作建议

### 作者建议的未来工作
1. **单模型融合**：将高层和低层合并到一个VLM中，推理时动态切换System 1/System 2模式
2. **多层级异步推理**：同时在不同抽象层级上异步处理输入和语言，而非固定频率
3. **跨层感知**：让高层了解低层的执行成功/失败情况，形成闭环改进

### 基于分析的未来方向
1. **自动化合成数据管线的去prompt-engineering化**：让合成数据生成本身也成为学习过程
2. **Hi Robot + π₀.₅**：将5类异构数据源的泛化能力注入低层VLA，进一步提升物理泛化
3. **Hi Robot + RTC**：用RTC优化低层的实时性能，使层级系统在动态任务中更鲁棒
4. **跨层联合训练**：高层和低层端到端联合训练（高层loss也考虑低层执行结果）
5. **多模态反馈**：扩展到非语言反馈（手势、示范），进一步增强人机交互的灵活性

## 我的综合评价

### 价值评分

#### 总体评分
**9.0/10** - 系统性地解决了VLA在复杂语言交互中的根本缺陷。层级架构+合成数据的组合设计优雅实用。评估全面（3平台×3领域×20 trials），定性分析深入。是对"如何让机器人真正理解人"这一问题的重大推进。

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | System 1/2的VLM实例化+反向合成数据生成，两者皆为新颖且实用的贡献 |
| 技术质量 | 9/10 | 架构清晰，合成数据方案可复现，消融实验严谨（hierarchy vs flat, syn vs no-syn） |
| 实验充分性 | 9/10 | 3平台×3任务×多基线+多消融+blind human eval+20 trials/condition |
| 写作质量 | 9/10 | 问题动机充分，System 1/2类比直觉好，定性对比图清晰展示失败模式 |
| 实用性 | 8/10 | 模块化设计实用，但合成数据仍需要prompt engineering |

### 重点关注

#### 值得关注的技术点
- 合成数据反向生成方案（observation → 假设prompt，而非prompt → observation）
- 高层和低层使用相同backbone但不同fine-tuning——架构统一性
- 高层推理的触发策略（1秒间隔 + 事件驱动）——简单但有效
- 语言作为层间接口——可解释、可debug、模块化

#### 需要深入理解的部分
- 合成数据的质量和多样性如何影响高层泛化
- 两层之间的语言接口是否存在"信息压缩"损失（高层观察到的图像信息低层也能看到吗？）
- 如何确定"恰到好处的子任务粒度"（太细→高层负担重，太粗→低层可能无法执行）

## 相关论文

### 直接相关
- [[pi0_Vision-Language-Action_Flow_Model|π₀]] - 低层VLA的基础架构
- YAY Robot (Shi et al., 2024) - 前序工作，语言交互修正
- SayCan (Brohan et al., 2023) - VLM+预定义技能的层级方案
- PaLM-E (Driess et al., 2023) - 多模态VLM用于机器人控制

### 背景相关
- [[pi0.5_Open_World_Generalization|π₀.₅]] - 互补的物理泛化
- [[KI_Knowledge_Insulation_for_VLA|KI]] - 训练基础设施
- [[RTC_Real_Time_Action_Chunking|RTC]] - 可叠加的实时推理优化
- [[pi0.6_RECAP_VLA_Learns_From_Experience|π₀.₆ RECAP]] - 可叠加的自改进能力

### 后续工作
- 尚无公开的后续论文（本文发表于ICML 2025）

## 外部资源
- [项目页面](https://pi.website/research/hirobot) - 含视频演示
- Gemini Robotics (2025) - 同样使用了分层System 1/System 2设计理念

> [!tip] 关键启示
> **把"想"和"做"分开——用两个VLM实现System 1/System 2架构。** 高层VLM负责理解复杂语言（"做一个素食三明治但不要番茄"）并输出简单命令（"拿起生菜"），低层VLA负责将简单命令转化为精确物理动作。两者通过语言接口连接——可解释、可升级、各司其职。合成数据反向生成（给定技能→生成假设prompt）是让高层学会处理多样化语言的关键。

> [!warning] 注意事项
> - 两层训练分离——高层不知道低层的实际能力边界，可能发出不可执行的命令
> - 合成数据生成仍依赖prompt engineering——自动化和通用性是未来方向
> - 高层推理引入额外延迟（1秒间隔），对极高频任务可能有影响
> - 评估依赖人工评分——大规模迭代优化成本高
> - 当前系统要求用户主动发出语言指令——被动场景（隐式意图理解）未覆盖

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！本文是层级VLA设计的标杆工作。对于致力于让机器人理解人类复杂指令的研究者和实践者，Hi Robot提供了一个完整、可复现、经过充分验证的方案。论文中对GPT-4o失败模式的定性分析尤其有启发意义——揭示了通用VLM在物理交互场景中的根本局限性。
