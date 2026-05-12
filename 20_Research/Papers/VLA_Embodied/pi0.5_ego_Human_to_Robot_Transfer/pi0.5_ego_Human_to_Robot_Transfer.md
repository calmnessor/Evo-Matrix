---
date: "2026-05-12"
paper_id: "arXiv:2512.22414"
title: "Emergence of Human to Robot Transfer in Vision-Language-Action Models"
authors: "Simar Kareer, Karl Pertsch, James Darpinian, Judy Hoffman, Danfei Xu, Sergey Levine, Chelsea Finn, Suraj Nair"
domain: "VLA_Embodied"
tags:
  - 论文笔记
  - VLA_Embodied
  - VLA
  - Human-to-Robot-Transfer
  - Cross-Embodiment
  - Emergence
  - Egocentric-Data
  - Co-Training
  - π0-series
quality_score: "8.8/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# Emergence of Human to Robot Transfer in Vision-Language-Action Models (π₀.₅+ego)

## 核心信息
- **论文ID**：arXiv:2512.22414
- **作者**：Simar Kareer (PI + GaTech), Karl Pertsch (PI), James Darpinian (PI), Judy Hoffman (GaTech), Danfei Xu (GaTech), Sergey Levine (PI), Chelsea Finn (PI), Suraj Nair (PI)
- **机构**：Physical Intelligence & Georgia Institute of Technology
- **发布时间**：2025-12-27
- **会议/期刊**：--
- **链接**：[arXiv](https://arxiv.org/abs/2512.22414) | [PDF](https://arxiv.org/pdf/2512.22414)
- **引用**：--

## 摘要翻译

### 英文摘要
Vision-language-action (VLA) models can enable broad open world generalization, but require large and diverse datasets. It is appealing to consider whether some of this data can come from human videos, which cover diverse real-world situations and are easy to obtain. However, it is difficult to train VLAs with human videos alone, and establishing a mapping between humans and robots requires manual engineering and presents a major research challenge. Drawing inspiration from advances in large language models, where the ability to learn from diverse supervision emerges with scale, we ask whether a similar phenomenon holds for VLAs that incorporate human video data. We introduce a simple co-training recipe, and find that human-to-robot transfer emerges once the VLA is pre-trained on sufficient scenes, tasks, and embodiments. Our analysis suggests that this emergent capability arises because diverse pretraining produces embodiment-agnostic representations for human and robot data.

### 中文翻译
VLA模型可实现广泛的开放世界泛化，但需要大量多样化的数据集。一个自然的想法是从人类视频中获取部分数据——人类视频覆盖了多样的真实世界场景且易于获取。然而，仅用人类视频训练VLA很困难，建立人类与机器人之间的映射需要人工工程，是一个重大研究挑战。受大语言模型的启发——其中从多样化监督中学习的能力随规模涌现——本文探究类似现象是否适用于融合人类视频数据的VLA。本文提出一种简单的协同训练方案，发现当VLA在足够多的场景、任务和embodiment上预训练后，人机迁移能力会涌现出来。分析表明，这种涌现能力源于多样化预训练产生了对人类和机器人数据的embodiment无关表征。

### 核心要点提炼
- **研究背景**：人类视频数据丰富且易于获取，但直接用于VLA训练受限于人机域差异（视觉、运动学）
- **研究动机**：受LLM涌现能力启发，探究人机迁移是否也是VLA预训练多样性的涌现特性
- **核心方法**：将人类视为"另一种embodiment"，用与机器人数据相同的训练目标（流匹配动作预测+子任务语言预测），以50:50比例与最相关机器人任务协同微调
- **主要结果**：在4个泛化基准上，人机迁移随预训练多样性增加而涌现（如Spice 32%→71%，Sort Eggs 57%→78%）；人类数据在2/3任务上接近目标机器人数据的微调效果
- **研究意义**：为人机迁移提供了一个新视角——不需要专门的显式对齐算法，只需将人类数据视为跨embodiment训练的一种形式，其效果随预训练规模而放大

## 研究背景与动机

### 领域现状
VLA模型通过异构监督（机器人遥操作、VLM数据、语言标注）实现泛化操作。然而，机器人遥操作数据获取成本高，而人类视频数据（特别是带3D手部追踪的egocentric视频）覆盖丰富的真实场景且易于大规模获取。近年工作（EgoMimic、EgoBridge、EMMA等）尝试直接从人类视频学习策略，但通常需要显式对齐（运动学、视觉、潜在空间），且在小规模下表现脆弱。

### 现有方法的局限性
1. **显式对齐方法不通用**：通过在机器人上叠加AR/VR或手工设计映射来对齐人机动作，限制了可捕捉的任务类型
2. **小规模下迁移脆弱**：基于少量机器人数据的人类视频训练往往失败，缺乏对"何时能work"的理解
3. **缺乏涌现视角**：先前工作聚焦于设计更好的对齐算法，而忽视了一个根本问题——人机迁移是否像LLM能力一样，是预训练规模的涌现特性？

### 研究动机
核心问题：**人类视频数据中的技能学习（无显式对齐）是否随VLA预训练规模而涌现？**

## 研究问题

### 核心研究问题
1. 人机迁移是否需要显式对齐算法？还是可以仅通过协同训练+多样化预训练自然产生？
2. 人机迁移能力如何随预训练多样性变化？（涌现曲线）
3. 迁移发生在什么层面？（高层语义/子任务 vs 低层动作）
4. 人类数据的价值与跨embodiment机器人数据相比如何？
5. 涌现能力的表征层面证据是什么？

## 方法概述

### 核心思想
**将人类视为"另一种embodiment"**——在VLA训练中将人类数据与机器人数据用完全相同的方式处理，不做任何显式对齐。核心假设是：足够多样化的预训练会使VLA自然形成embodiment无关的表征，从而使人类数据的技能迁移自发涌现。

### 方法框架

#### 整体架构

![[arch.jpg|800]]

> 图1：π₀.₅+ego架构。基于π₀.₅ VLA架构，使用PaliGemma VLM骨干网络+action expert，同时训练FAST离散token（自回归loss）和连续动作（流匹配loss）。人类数据使用与机器人数据完全相同的训练目标。

#### 人类数据采集管线

![[apparatus.jpg|800]]

> 图2：人类数据采集设备。头戴式高分辨率相机+双手腕相机。通过视觉SLAM重建头部相机的6D位姿和双手21个3D关键点。用掌心+中指+无名指定义"手部末端执行器"位姿。

**数据采集**：
- 设备：头戴式高分辨率相机 + 可选双手腕相机（共3个同步视频流）
- 采集方式：仿照episodic遥操作——操作员在目标场景中反复执行任务演示
- 数据量：Bussing 3h、Spice 3h、Dresser 3h、Sort Eggs 5h（共14小时）

**动作空间定义**：
- 机器人动作：$a \in \mathbb{R}^{H \times 16}$（左右臂各6DoF+gripper + 2维底盘动作）
- 人类动作：$a \in \mathbb{R}^{H \times 18}$（左右手各6DoF + 6维头部相机位姿变化作为"底盘"代理，无gripper动作）
- 动作块长度：$H=50$

#### 训练目标

使用π₀.₅/KI的两阶段训练方案，人类数据上使用两个目标：

1. **低层动作预测**：FAST离散token的自回归交叉熵loss + 连续动作的流匹配loss
2. **高层子任务预测**：给定观测和语言指令，预测子任务语言描述 $\pi_\theta(l^\text{subtask}_t | o_t, l_t)$

#### 训练混合比例

关键设计：**人类数据与最相关的机器人任务数据按50:50比例混合协同微调**。

![[data.jpg|800]]

> 图3：数据构成与泛化基准。4个任务分别测试场景（Spice, Dresser）、物体（Bussing）和任务（Sort Eggs）三个维度的泛化。

### 与KI方法的关系

本方法直接继承KI（Knowledge Insulation）的训练方案：
- VLM骨干网用FAST离散token学习表征（训练时）
- Action expert用流匹配生成连续动作（推理时）
- Stop-gradient保护VLM预训练知识
- 在此基础上将人类数据作为额外embodiment加入训练混合

## 实验结果

### 实验目标
验证：（1）π₀.₅+ego能否实现人机迁移；（2）迁移是否随预训练多样性涌现；（3）迁移发生在什么层面；（4）人类数据与机器人数据的相对价值

### 泛化基准

| 任务 | 泛化维度 | 机器人数据覆盖 | 人类数据新增 | 指标 |
|------|---------|---------------|-------------|------|
| Spice | 场景（新厨房） | 固定厨房 | 目标厨房 | 二值成功率 |
| Dresser | 场景（新卧室） | 固定卧室 | 目标卧室 | 二值成功率 |
| Bussing | 物体（新物品） | 餐具+垃圾 | 厨房工具 | 正确放置物品数 |
| Sort Eggs | 任务（新语义） | 拿鸡蛋放蛋盒 | 按颜色分类放入不同蛋盒 | 正确分类数 |

### 主要结果

#### 人机迁移总体效果

![[taskPerf.jpg|800]]

> 图4：四个泛化任务的迁移结果。所有任务上人类数据协同训练均显著提升泛化性能。

| 任务 | π₀.₅ (仅机器人) | π₀.₅+ego (机器人+人类) | 提升 |
|------|-----------------|----------------------|------|
| Spice | 32% | 71% | +122% |
| Dresser | 25% | 50% | +100% |
| Bussing | 53% | 63% | +19% |
| Sort Eggs | 57% sorting | 78% sorting | +37% |

**Sort Eggs尤为突出**：纯机器人数据训练的模型只能随机放置鸡蛋（57%的分类准确率），加入人类分类演示后分类准确率达78%，且平均多正确放置4个鸡蛋。

#### 核心发现：人机迁移随预训练多样性涌现

![[scalingAllLines.jpg|800]]

> 图5：人机迁移作为预训练多样性的函数。x轴为预训练多样性，黄色和蓝色分别为有/无人类数据。关键观察：在低多样性（0%, 25%）下无迁移增益，迁移在75%+才涌现，100%+跨embodiment预训练进一步放大。

**预训练初始化级别**：
- `0%`：base VLM初始化（PaliGemma）
- `25%, 50%, 75%, 100%`：在目标embodiment（ARX + mobile ARX）的[场景×任务]组合上逐渐增加的预训练
- `100% + X-emb`：π₀.⑤的完整预训练混合（含大量非目标embodiment数据）

![[scalingEggs.jpg|800]]

> 图6：Sort Eggs任务的scaling趋势。纯机器人数据性能即使增加预训练也趋于平台（57%），但人类数据带来的增益随预训练大幅增长。

**关键洞察**：
- 某些情况下，增加预训练多样性不提升zero-shot泛化，但大幅提升对人数据的迁移能力（如Spice 50%→75%的zero-shot仅小幅改善，但迁移增益骤增）
- 这说明预训练的"泛化能力"和"迁移能力"可能是不完全相关的两个属性
- 存在一个"临界多样性"，超过后人机迁移才可能发生

#### 表征分析：Embodiment无关表征随预训练涌现

![[tsne.jpg|800]]

> 图7：T-SNE可视化。左：低预训练多样性下，人类和机器人的VLA输出embedding完全分离（互不重叠的cluster）；右：高预训练多样性下，两者表征逐渐重叠（统一的embodiment无关表征）。

分析方式：将人类和机器人观测输入协同微调后的VLA，取前200个输出embedding的mean pooling（大致对应"任务"表征），用T-SNE可视化。

**结论**：先前在小数据上的工作发现协同训练后人类和机器人表征仍分离（需要显式对齐），而本文发现**充分的预训练多样性使协同训练本身就能产生对齐的表征**——不需要任何显式对齐方法。

#### 人类数据 vs 机器人数据

![[target.jpg|800]]

> 图8：人类数据与目标机器人数据对比。在Sort Eggs和Dresser上人类数据微调与目标机器人数据微调效果相当；Bussing上机器人数据更有优势（25% vs 65%）。

![[ur5.jpg|800]]

> 图9：人类数据 vs 跨embodiment机器人数据（UR5→ARX）。两者提供相似水平的迁移——都超越基线但都不及目标机器人数据，说明人类数据迁移与跨embodiment机器人迁移共享类似的性质。

#### 高层 vs 低层迁移

![[hl_ll.jpg|800]]

> 图10：高层（HL subtask）vs 低层（LL action）迁移消融。仅在单一层面使用人类数据效果不如同时在两层使用——迁移同时发生在两个层面。

关键失败模式：
- **仅HL迁移**：LL policy不遵循正确的子任务命令（如"拿起香料瓶"被误解为拿起已经在托盘上的瓶子）
- **仅LL迁移**：HL policy发出错误指令或重复指令（如持续预测"拿起香料瓶"即使瓶子已被拿起）

#### 手腕相机的价值

![[wrist.jpg|800]]

> 图11：手腕相机消融。Bussing和Dresser任务受益于腕部相机的额外观测；Spice和Eggs任务受影响不大。建议进行大规模人类数据采集时配备腕部相机以最大化覆盖任务空间。

## 深度分析

### 研究价值评估

#### 理论贡献
- **贡献1：发现人机迁移的涌现（emergence）性质**
  - 创新点：首次系统性证明人机迁移不是方法设计问题，而是预训练规模问题——在足够多样化预训练下**自然涌现**
  - 学术价值：改变了人们思考人机迁移的方式——从"设计更好的对齐算法"转向"扩大预训练多样性"
  - 影响范围：VLA训练、跨embodiment迁移、人类数据利用

- **贡献2：表征层面解释**
  - 创新点：通过T-SNE可视化证明多样化预训练导致embodiment无关表征的自然形成
  - 学术价值：为涌现现象提供了可解释的证据

- **贡献3：简单有效的方法论**
  - 创新点：零显式对齐——仅将人类作为额外embodiment加入标准VLA训练混合
  - 学术价值：提供了可复现的方法论基线

#### 实际应用价值
- **应用场景1：快速在新场景中部署机器人**
  - 适用性：只需在新目标场景采集少量人类视频（10小时量级），无需部署机器人
  - 优势：人类视频采集远比机器人遥操作方便快捷
  - 潜在影响：大幅降低机器人部署到新环境的数据成本

- **应用场景2：教授机器人新任务语义**
  - 适用性：Sort Eggs类——机器人已有基本操作技能，需新增任务语义/规则
  - 优势：人类视频提供自然的新任务演示

#### 领域影响
- **短期影响**：为VLA训练数据策略提供了一个经过验证的方向——投资预训练多样性以放大后续人类数据的价值
- **中期影响**：人类视频数据可能成为VLA训练的标准数据源之一
- **长期影响**：随着更大规模被动人类视频（非episodic）的获取，人机迁移可能进一步放大

### 方法优势详解

#### 优势1：极简方法论
- **描述**：不需要任何显式对齐（运动学映射、潜在空间对齐、域适应等），仅标准VLA训练目标
- **技术基础**：依赖预训练模型的embodiment无关表征能力，而非手工设计的对齐
- **对比**：与EgoBridge、EgoMimic等需要显式对齐的方法形成鲜明对比

#### 优势2：数据效率
- **描述**：仅需10-20小时人类视频即可实现显著迁移
- **技术基础**：预训练阶段的大量数据已经建立了embodiment无关的基础表征
- **对比**：相比从头训练的方法，数据效率极高

#### 优势3：双向迁移
- **描述**：迁移同时发生在高层（子任务语义理解）和低层（动作执行方式）
- **技术基础**：两者都作为训练目标，模型自然地同时在两个层面学习对齐

### 局限性分析

#### 局限1：人类数据量仍然有限
- **描述**：仅在4个任务上各采集了3-5小时episodic数据（总计14小时）
- **表现**：未探索更大规模被动人类视频的利用
- **原因**：episodic采集方式仍有成本
- **可能方案**：扩展到真正的大规模被动egocentric视频（如Ego4D）

#### 局限2：Bussing任务人类数据明显弱于目标机器人数据
- **描述**：Bussing上人类数据（25%）远不如目标机器人数据（65%）
- **原因**：可能是因为操作精度要求高，或人类手部动作与机器人夹爪的差异在此任务中无法被忽略
- **影响**：对于需要精确gripper操作的任务，纯人类数据可能不足以替代机器人数据

#### 局限3：无gripper动作
- **描述**：人类数据不包含"夹爪开合"动作（难以从手部关键点估计），仅依赖机器人数据学习
- **原因**：手部张开程度难以从视觉关键点精确估计
- **影响**：对于需要复杂夹爪控制的任务，可能存在瓶颈

#### 局限4：尚未在预训练阶段使用人类数据
- **描述**：当前仅在微调阶段使用人类数据，而未在预训练中引入
- **可能方案**：大规模的被动egocentric视频可能在预训练阶段就提供价值

### 适用性与场景分析

#### 适用场景
- **需要在新场景快速部署**：已有基础VLA，只需采集目标场景人类视频即可泛化
- **教授新任务语义**：机器人已有基本操作技能（如pick-place），需学会新规则（如sorting）
- **难以部署机器人进行遥操作的场景**：人类视频采集门槛远低于机器人遥操作

#### 不适用场景
- **需要极高操作精度的任务**：纯人类数据可能无法提供足够的gripper精度信息
- **缺乏多样化预训练VLA**：方法依赖于足够强大的预训练VLA
- **与现有机器人数据域差异极大**：如果新场景与所有已有机器人数据都差异巨大，迁移可能困难

## 与相关论文对比

### [[pi0.5_Open_World_Generalization|π₀.₅]] - Open-World Generalization

#### 方法对比
| 对比维度 | π₀.₅ | 本文（π₀.₅+ego） |
|----------|-------|-------------------|
| 核心思想 | 异构数据协同训练实现开放世界泛化 | 在π₀.₅基础上新增人类embodiment |
| 数据源 | 5类（MM, ME, CE, HL, WD） | π₀.₅数据 + 人类egocentric视频 |
| 训练目标 | 自回归FAST + 流匹配 | 与π₀.₅相同，人类数据使用同样目标 |
| 泛化方式 | 数据多样性驱动 | 预训练多样性涌现→人类数据迁移 |

**关系分析**：直接基于π₀.₅架构和训练方案，核心贡献是证明人类数据可以作为π₀.₅的第6种数据源，且其效果随前5种数据的多样性而涌现。

### [[KI_Knowledge_Insulation_for_VLA|KI (Knowledge Insulation)]] - Train Fast, Run Fast

#### 方法对比
| 对比维度 | KI | 本文 |
|----------|-----|------|
| 核心贡献 | Stop-gradient + joint training | 人机迁移涌现发现 |
| 架构 | π₀ + KI training recipe | 复用KI架构 |
| 数据 | VLM数据协同训练 | VLM数据 + 人类视频 |
| 焦点 | 保护VLM知识不被破坏 | 用保护好的VLM知识消化新embodiment |

**关系分析**：KI提供了训练基础设施（stop-gradient等），使VLM知识在微调时不被破坏，从而使人机迁移得以发生——可以理解为KI是本文方法的技术前提。

### [[pi0.6_RECAP_VLA_Learns_From_Experience|π₀.₆ RECAP]] - VLA Learns From Experience

**关系分析**：π₀.₅+ego和π₀.₆ RECAP分别从不同方向探索VLA的能力扩展：
- **本文**：横向扩展——从新数据源（人类视频）学习新技能
- **RECAP**：纵向改进——通过RL在已有任务上自我优化

两者互补：未来可结合（用人类视频学习新任务→再用RL自我改进）。

### EgoMimic / EgoBridge

#### 方法对比
| 对比维度 | EgoMimic/EgoBridge | 本文 |
|----------|-------------------|------|
| 对齐方式 | 显式（潜在空间对齐、视觉对齐等） | 零显式对齐（依赖预训练多样性） |
| 数据规模 | 小规模 | 大规模预训练 + 少量人类微调 |
| 通用性 | 较低（对齐方法为特定任务设计） | 较高（同方法适用于所有任务） |

**关系分析**：本文与显式对齐方法形成方法论对比——不需要设计复杂对齐算法，只需要充分的预训练多样性。这是对领域思维方式的根本性转变。

## 技术路线定位

### 所属技术路线
本文属于**跨embodiment迁移学习**技术路线，具体停留在"人类作为一种embodiment"的交叉点。其独特贡献是为该路线提供了**涌现（emergence）视角**。

### 技术路线发展历程

```
人工设计对齐（EgoMimic等）→ 潜在空间对齐（EgoBridge）→ 涌现对齐（本文）
         ↓                          ↓                        ↓
   手工规则+小规模            需要对齐算法+中规模        零显式对齐+大规模预训练
```

### 本文在技术路线中的位置
- **承上**：继承π₀系列架构、KI训练recipe、跨embodiment训练的思想
- **启下**：为大规模被动人类视频数据的利用铺路；提出"涌现"视角指导未来方向
- **关键节点**：范式转变——从"如何对齐"转向"何时自然对齐"

## 未来工作建议

### 作者建议的未来工作
1. 将人类数据引入**预训练阶段**（而非仅微调），利用大规模被动egocentric视频
2. 探索其他可能随规模涌现的能力

### 基于分析的未来方向
1. **人类数据预训练（Human Pre-training）**：将大规模egocentric数据集（如Ego4D的扩展版）直接纳入VLA预训练混合，预期能进一步提升迁移效果
2. **更精确的手部动作提取**：通过手部mesh重建等技术改进gripper动作的近似，使人类数据在需要精确操作的任务上也接近目标机器人数据的价值
3. **涌现阈值研究**：精确量化"临界多样性"——多少数据集规模/场景数/任务数/embodiment数才能触发迁移涌现？
4. **被动视频利用**：本工作使用episodic人类演示，未来可扩展到真正的"in-the-wild"被动视频（人们日常做家务的录像）

## 我的综合评价

### 价值评分

#### 总体评分
**8.8/10** - 提出了一个范式级的视角转变（人机迁移是涌现现象），实验设计严谨（多轴泛化基准+多级预训练消融+表征分析），方法论极简。主要扣分在于人类数据规模有限（仅14h）和尚未在预训练中使用。

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | "人机迁移是涌现现象"的发现是范式级贡献，改变了领域对问题的思考方式 |
| 技术质量 | 8/10 | 方法简单但合理，消融实验全面。但缺少更深入的涌现理论分析 |
| 实验充分性 | 9/10 | 4任务×3泛化维度×6级预训练×多组消融，实验矩阵全面 |
| 写作质量 | 9/10 | 论文叙事清晰，类比LLM涌现很有说服力，图表质量高 |
| 实用性 | 8/10 | 方法极简可复现，但依赖大规模预训练VLA（门槛高） |

### 重点关注

#### 值得关注的技术点
- 人类动作空间定义（掌心+中指+无名指→末端执行器位姿）
- 50:50的微调混合比例对迁移效果的影响
- TSNE embedding分析——前200个输出token的mean pooling
- 预训练的"泛化"和"迁移"能力可能不相关——这对理解VLA能力很重要

#### 需要深入理解的部分
- 为什么人类数据在Bussing任务上效果明显不如目标机器人数据？（精确gripper操作的缺失？）
- 涌现的"临界点"具体由什么因素决定？（场景数？任务数？embodiment数？）
- 如果人类数据也在预训练中使用，scaling curve会是什么样子？

## 相关论文

### 直接相关
- [[pi0.5_Open_World_Generalization|π₀.₅]] - 基础VLA模型，本文直接基于其架构和训练方案
- [[KI_Knowledge_Insulation_for_VLA|KI (Knowledge Insulation)]] - 训练基础设施（stop-gradient + joint training）
- [[pi0_Vision-Language-Action_Flow_Model|π₀]] - 基础架构

### 背景相关
- [[pi0.6_RECAP_VLA_Learns_From_Experience|π₀.₆ RECAP]] - 互补的VLA自我改进方法
- [[FAST_Efficient_Action_Tokenization_for_VLA|FAST]] - 动作token化
- EgoMimic - 先前的人机迁移工作（显式对齐）
- EgoBridge - 先前的人机迁移工作（潜在空间对齐）

### 后续工作
- 尚无公开的后续论文

## 外部资源
- [Physical Intelligence Blog](https://pi.website/) - 可能有相关博客
- Ego4D - 大规模egocentric视频数据集（潜在数据源）

> [!tip] 关键启示
> **人机迁移不需要复杂的对齐算法——它是预训练多样性的涌现现象**。就像LLM在足够规模后涌现出chain-of-thought能力一样，足够多样化的VLA预训练使模型自然形成embodiment无关的表征，从而使人类视频数据可以像另一种机器人embodiment数据一样被利用。这一发现将研究重心从"设计更好的对齐"转向"建立更多样化的预训练数据"。

> [!warning] 注意事项
> - 方法依赖大规模多样化预训练VLA——对小团队可能不直接适用
> - 人类数据仅14小时且为episodic（非in-the-wild），效果可能在更大规模上有变化
> - 无gripper动作近似——需要gripper控制的任务可能受益有限
> - Bussing任务上人类数据（25%）远不如目标机器人数据（65%），存在明显的能力上限
> - 涌现的精确阈值（需要多少场景/任务/embodiment）尚未被精确定量

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！本文是VLA研究的一个重要范式转变——它从根本上改变了我们对人机迁移问题的理解。对于从事VLA、跨embodiment迁移或人类视频学习的读者，文中关于涌现的发现和表征层面的分析都值得仔细研读。
