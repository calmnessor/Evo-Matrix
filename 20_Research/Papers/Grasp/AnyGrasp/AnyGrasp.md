---
date: "2022-12-16"
paper_id: "arXiv:2212.08333"
title: "AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains"
authors: "Hao-Shu Fang, Chenxi Wang, Hongjie Fang, Minghao Gou, Jirong Liu, Hengxu Yan, Wenhai Liu, Yichen Xie, Cewu Lu"
domain: "Grasp"
tags:
  - 论文笔记
  - Grasp
  - 抓取感知
  - 3D点云
  - 动态抓取
  - 7-DoF
  - GraspNet
  - GSNet
  - Temporal-Tracking
quality_score: "8.7/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains

## 核心信息
- **论文ID**：arXiv:2212.08333
- **作者**：Hao-Shu Fang, Chenxi Wang, Hongjie Fang, Minghao Gou, Jirong Liu, Hengxu Yan, Wenhai Liu, Yichen Xie, Cewu Lu
- **机构**：上海交通大学电子信息与电气工程学院 (MVIG Lab, 卢策吾组)
- **发布时间**：2022-12-16 (v1), 2023-06-06 (v2)
- **会议/期刊**：IEEE Transactions on Robotics (T-RO)
- **链接**：[arXiv](https://arxiv.org/abs/2212.08333) | [PDF](https://arxiv.org/pdf/2212.08333) | [Project](https://graspnet.net/anygrasp.html)
- **引用**：IEEE T-RO accepted

## 摘要翻译

### 英文摘要
As the basis for prehensile manipulation, it is vital to enable robots to grasp as robustly as humans. Our innate grasping system is prompt, accurate, flexible, and continuous across spatial and temporal domains. Few existing methods cover all these properties for robot grasping. In this paper, we propose AnyGrasp for grasp perception to enable robots these abilities using a parallel gripper. Specifically, we develop a dense supervision strategy with real perception and analytic labels in the spatial-temporal domain. Additional awareness of objects' center-of-mass is incorporated into the learning process to help improve grasping stability. Utilization of grasp correspondence across observations enables dynamic grasp tracking. Our model can efficiently generate accurate, 7-DoF, dense, and temporally-smooth grasp poses and works robustly against large depth-sensing noise. Using AnyGrasp, we achieve a 93.3% success rate when clearing bins with over 300 unseen objects, which is on par with human subjects under controlled conditions. Over 900 mean-picks-per-hour is reported on a single-arm system. For dynamic grasping, we demonstrate catching swimming robot fish in the water.

### 中文翻译
作为抓取操控的基础，赋予机器人像人类一样鲁棒抓取的能力至关重要。人类的先天抓取系统快速、准确、灵活，且在空间和时间域上具有连续性。现有抓取方法很少能同时覆盖所有这些特性。本文提出 AnyGrasp 抓取感知系统，使用平行夹爪使机器人具备这些能力。具体而言，我们开发了一种时空域上的密集监督策略，结合真实感知和解析标注。在训练过程中引入对物体质心的感知以提升抓取稳定性，并利用观测间的抓取对应关系实现动态抓取跟踪。模型可高效生成准确、7-DoF、密集且时间平滑的抓取姿态，并对深度感知噪声具有强鲁棒性。使用 AnyGrasp，在包含 300+ 未见物体的清箱实验中取得了 93.3% 成功率，与受控条件下的人类表现相当。单臂系统可达 900+ 次/小时的平均拾取速度。对于动态抓取，展示了在水中抓取游动机器鱼的实验。

### 核心要点提炼
- **研究背景**：机器人抓取需要在空间（密集覆盖）和时间（连续跟踪）两个维度上达到人类水平
- **研究动机**：现有方法要么只能处理静态场景，要么在速度-精度间做权衡，缺乏统一的时空抓取方案
- **核心方法**：几何处理模块（密集7-DoF抓取姿态预测）+ 时序关联模块（抓取对应对匹配），融入质心感知
- **主要结果**：300+未见物体 93.3% 成功率（追平人类），900+ MPPH，动态抓鱼 75.5% 成功率
- **研究意义**：首个统一的时空连续7-DoF抓取感知系统，证明少量真实数据训练可超越大规模仿真数据

## 研究背景与动机

### 领域现状
视觉引导的机器人抓取是机器人领域的核心问题。早期方法假设已知物体的完整3D模型，基于学习的方法通过大规模数据和自动特征提取缓解了这一问题。表示方法从点表示、点对表示、矩形表示逐步演进到6-DoF抓取姿态检测。近期，Fang 等人提出基于端到端网络的全局推理方法（GraspNet），Wang 等人提出 graspness 机制进一步筛选不可行抓取姿态。

### 现有方法的局限性
1. **采样-评估范式**（如 GPD, DexNet 4.0）：需要在计算时间和生成抓取数量间权衡，通常耗时数秒且无法密集覆盖
2. **端到端方法**（如 GraspNet, S4G）：主要关注静态场景，未处理动态抓取
3. **动态抓取方法**：要么需要物体先验信息，要么仅在图像坐标系中跟踪最近抓取姿态，无法保证物体坐标系下的一致性
4. **仿真训练方法**：sim-to-real gap 严重，在低成本深度传感器上性能大幅下降

### 研究动机
人类抓取系统具备四大特性：快速（<100ms）、准确、灵活、时空连续。机器人抓取系统应同时具备这些能力，但此前没有任何方法覆盖全部维度。此外，极少工作深入分析训练数据中真实数据 vs 仿真数据、标注密度、场景多样性等因素的影响。

## 研究问题

### 核心研究问题
**如何构建一个统一的抓取感知系统，在空间和时间两个维度上同时实现快速、准确、7-DoF、密集且时间连续的抓取姿态检测？**

具体包括：
1. 如何在单次前向传播中生成密集覆盖全场景的 7-DoF 抓取姿态
2. 如何在多帧观测间建立抓取姿态的对应关系，实现时间平滑跟踪
3. 如何在仅使用少量真实物体（144个）训练的情况下，在超过300个未见物体上达到人类水平
4. 如何让机器人抓取系统鲁棒应对不同的深度传感器噪声

## 方法概述

### 核心思想
AnyGrasp 的核心思想是**以密集空间预测为基础，赋能时间连续跟踪**：几何处理模块在单帧点云上一次性预测密集的 7-DoF 抓取姿态（空间连续），时序关联模块为每个抓取姿态生成特征向量，通过余弦相似度匹配跨帧的抓取对应关系（时间连续）。此外，融入物体质心感知提升抓取稳定性，让网络隐式学习碰撞避免。

### 方法框架

#### 整体架构

![[model.pdf|800]]

> 图1：AnyGrasp 模型架构。对于输入的部分点云，几何处理模块预测密集抓取姿态。时序关联模块为 M 个预测抓取姿态各生成一个 C 维特征向量。该特征向量以抓取姿态对在物体坐标系中的距离为目标学习：距离越近，余弦相似度越高。

详细结构：

![[smallmodel.pdf|400]]

> 图2：几何处理模块的详细结构。包含 Backbone、MLP、Graspable FPS、Graspable PVS、Cylinder Grouping 等组件。

#### 各模块详细说明

**模块1：几何处理模块（Geometry Processing Module）**

基于 GSNet（Wang et al.），以场景部分点云 $\mathcal{P}$ 为输入：
- **Backbone**：3D 卷积网络提取逐点几何特征
- **MLP-1**：生成 objectness mask 和 graspable 热力图
- **Graspable FPS**：根据 objectness + graspable 概率采样 $\mathcal{M}=1024$ 个种子点
- **MLP-2**：为每个种子点预测 300 个视角得分，通过 Graspable PVS 选择最佳视角
- **Cylinder Grouping**：沿最佳视角在圆柱空间内聚合局部几何特征
- **MLP-3**：预测 12 个面内旋转 × 5 个接近深度（原始4个+新增0.5cm）的抓取得分和宽度，以及 12 个稳定得分
- **后处理**：将抓取得分乘以 $(1-\text{stable\_score})$，再参数化为 7-DoF 抓取姿态 $\mathcal{G} = [\mathbf{R}\ \mathbf{t}\ w]$

> 原始 GSNet 输出 $12\times4\times2$ 个值，AnyGrasp 修改最后一层输出 $12\times5\times2+12$ 个值，增加的12个值为稳定得分

**模块2：时序关联模块（Temporal Association Module）**

- **特征构建**：对每个种子点，拼接四类特征：
  1. **Seed features**：MLP-2 之前的几何特征
  2. **Grasp features**：MLP-3（最后一层）之前的抓取特征
  3. **Color features**：沿抓取方向的圆柱空间内 $\mathcal{K}=16$ 个点的 RGB 信息，经 MLP+Pooling 提取
  4. **Grasp pose parameters**：旋转矩阵(9) + 平移(3) = 12 维
- **融合**：拼接后经 MLP 生成 $\mathcal{C}=256$ 维特征向量
- **对应矩阵**：跨帧特征向量间的余弦相似度构成 $\mathcal{M}\times\mathcal{M}$ 对应矩阵
- **跟踪**：选定上一帧的 $n$ 个目标抓取，在当前帧的 $\mathcal{M}$ 个候选中选 top-$n$ 对应得分最高者

**模块3：损失函数**

几何处理模块沿用 GSNet 的 softmax 损失（objectness 分类）和 smooth-$l_1$ 损失（graspable 热力图和抓取参数回归）。

时序关联模块采用**有监督对比学习**：
$$L = \sum_{\mathcal{G}_i^1\in \mathbf{G}_1} \frac{-1}{|\mathbf{P}(i)|} \sum_{\mathcal{G}_k^2\in P(i)}\log \frac{\exp(s_\mathrm{corres}(\mathcal{G}_i^1,\mathcal{G}_k^2)/\tau)}{\sum_{\mathcal{G}_j^2\in \mathbf{G}_2}\exp(s_\mathrm{corres}(\mathcal{G}_i^1,\mathcal{G}_j^2)/\tau)}$$

其中 $\mathbf{P}(i)=\{\mathcal{G}_k^2\in \mathbf{G}_2 | d(\mathcal{G}_i^1,\mathcal{G}_k^2)\leq\sigma\}$ 表示与 $\mathcal{G}_i^1$ 属同一类的抓取姿态集合，$\sigma=0.1$，$\tau=0.1$。

**抓取距离度量**：将两个抓取姿态变换到物体坐标系后：
$$d(\mathcal{G}_1,\mathcal{G}_2) = \frac{\Delta\mathbf{t}}{w_{\max}} + \gamma\frac{\Delta\mathbf{R}}{\pi}$$

其中 $w_{\max}=0.01$m，$\gamma=0.1$。

### 关键设计

#### 稳定得分（Stable Score）
受人类抓取时倾向于关注物体质心的行为启发，定义稳定得分为归一化的夹爪平面到物体 COG 的垂直距离：

![[fig2.pdf|300]]

> 图3：夹爪平面到物体质心的垂直距离示意。假设夹爪在运输物体时会旋转至垂直姿态（与重力方向平行）。

$$s_{\text{stable}} = \frac{d_{\perp}(\text{gripper\_plane}, \text{COG})}{\max_{\mathcal{G}\in\text{object}} d_{\perp}(\text{gripper\_plane}, \text{COG})} \in [0,1]$$

最终抓取得分 = $\text{grasp\_score} \times (1-\text{stable\_score})$，得分越低表示抓取越不稳定。

#### 障碍物感知
网络以整个场景为输入，隐式学习障碍物避免——如果抓取姿态周围有障碍物且无法为夹爪预留空间，该姿态的抓取得分直接设为0。这避免了昂贵的显式碰撞检测。

#### 后处理
1. **碰撞检测**：对 top-100 抓取姿态进行基于点云的快速碰撞检测（将夹爪简化为三个立方体）
2. **夹爪居中**：计算两个指尖到接触点的距离，沿指尖连线方向平移夹爪确保两侧等距

| 组件 | 说明 | 耗时 |
|------|------|------|
| 几何处理模块 | 预测密集抓取姿态 | ~100ms |
| 碰撞检测+居中调整 | GPU 矩阵计算 | ~80ms |
| 总决策时间 | -- | <200ms |

## 实验结果

### 实验设置

#### 硬件
- **静态抓取**：UR5 机械臂 + Robotiq-85 夹爪 + 全局 D415/D435 深度相机
- **动态抓取**：Flexiv Rizon 臂 + L515 腕载相机 + Robotiq-85 加装 3D 打印延长夹爪
- **计算**：Intel i9-10900K + Nvidia RTX 2060 (Ubuntu 20.04)

#### 数据集
- **训练集**：扩展 GraspNet-1Billion（144 个真实物体，268 个场景，256 视角/场景）
- **测试集**：300+ 日常物体（超市/五金店/玩具店采购），涵盖硬件、零食、布偶、玩具、家居五大类
- **对抗测试集**：DexNet 2.0 的 13 个 + EGAD 的 49 个对抗物体

#### 基线方法
- **DexNet 4.0**：代表性采样-评估方法，在数千仿真物体上训练
- **Human**：人类操作员使用相同夹爪配置和开环策略
- **Nearest Tracking**：启发式动态跟踪基线（跟踪最近邻抓取姿态）

#### 评估指标
- **Attempt-Centric Success Rate**：成功抓取次数 / 总尝试次数
- **Object-Centric Success Rate (Completion Rate)**：成功抓取的物体数 / 总物体数
- **Mean Picks Per Hour (MPPH)**：平均每小时抓取次数

### 主要结果

#### 静态抓取实验结果

| 物体类别 | DexNet 4.0 (%) | **AnyGrasp (%)** | Human (%) |
|----------|---------------|-----------------|-----------|
| Hardware | 59.3 | **81.5** | 91.4 |
| Snack | 52.3 | **100.0** | 93.9 |
| Ragdoll | 87.4 | **100.0** | 96.6 |
| Toy | 72.8 | **93.1** | 91.8 |
| Household | 64.6 | **85.5** | 94.4 |
| **All** | 72.2 | **93.3** | 93.9 |

> 注：Object-Centric 指标下 AnyGrasp 达 99.8%，Human 100%，DexNet 4.0 为 98.9%

![[figure1a1.pdf|400]]

> 图4：不同物体类别上的抓取成功率对比。

#### 传感器鲁棒性

![[heatmap.pdf|300]]

> 图5：D415 与 D435 相机的深度偏差图及对应抓取成功率对比。D435 噪声更大（最高 $\pm5$mm），但 AnyGrasp 仍保持高成功率。

| 相机 | 成功率 |
|------|--------|
| D415 | 93.3% |
| D435 | 91.5% |

#### 对抗物体

![[figure1b.pdf|300]]

> 图6：对抗物体上的抓取成功率。AnyGrasp 显著优于 DexNet 4.0，但与人类仍有差距。

#### 碎片清理任务
成功清理破碎陶罐的薄片（厚度 < 3mm），展示了高精度抓取姿态估计的能力。

![[fragments.pdf|400]]

> 图7：机器人清理碎片的过程快照。需要精确估计薄片的抓取姿态。

#### 动态抓取：抓鱼实验

![[fish.pdf|300]]

![[vis.pdf|300]]

> 图8：不同机器鱼（左）和抓鱼实验表现（右）。AnyGrasp 平均成功率 75.5%，显著优于启发式最近邻方法的 62.5%。

![[catchfish.pdf|400]]

> 图9：机器人抓鱼过程：(a) 选择抓取目标；(b-e) 伺服跟踪；(e) 闭合夹爪；(f) 抓起。

| 方法 | 成功率 | 平均耗时 |
|------|--------|----------|
| **AnyGrasp** | **75.5%** | 基准 |
| Nearest Tracking | 62.5% | +12.7% |

#### 失败分析

![[fig7.pdf|400]]

> 图10：抓鱼实验的失败案例分析。

五类失败原因：
1. 鱼滑脱（~50%）：抓取姿态正确但水中摩擦太小
2. 预测抓取落在鱼前方
3. 抓取质量不够好，指尖推开了鱼
4. 预测抓取落在鱼后方（历史动量信息过时）
5. 对应切换：两条相似鱼靠太近导致跟踪 ID 混淆

#### 效率
- 推理：7 Hz (Nvidia RTX 2060)
- MPPH：900+（单臂 UR5，最大速度 1 m/s）
- 人类 MPPH：平均 1000-1200，最高 1500+

### 消融与讨论

#### 仿真训练 vs 真实数据训练

![[figure3d1.pdf|400]]

> 图11：真实数据训练与仿真数据训练（含高斯噪声增强）在真实机器人实验中的对比。

![[sim/ 2>&1 || echo "no sim subdir"; ls figure/sim/ 2>/dev/null]]

| 训练数据 | 测试集成功率 |
|----------|-------------|
| 真实数据 (144 objects) | **93.3%** |
| 仿真数据 + 高斯噪声 | 大幅下降 |
| 纯仿真数据 | 更差 |

关键发现：即使在仿真数据上使用高斯噪声增强的传统 sim-to-real 策略，性能仍然远不如直接使用真实数据训练。当剩余物体较少时，仿真训练的模型甚至无法生成有效的抓取姿态。

#### 稳定得分的影响
对长重型物体的实验：不考虑稳定得分时出现 16 次滑动（含 3 次失败），考虑后降至 11 次（含 2 次失败）。

#### 密集监督策略分析

![[fig9.pdf|600]]

> 图12：不同降采样维度下的模型性能。pose/image/scene 分别降采样 10x 和 50x。

关键发现：
- 抓取姿态密度 ↓10x ≈ 训练图像数量 ↓10x 的性能退化
- 场景多样性（）下降带来的性能退化最严重——降到只有 2 个场景时模型无法收敛
- **密集标注 ≈ 多样图像 >> 场景数量的重要性**

## 深度分析

### 研究价值评估

#### 理论贡献
- **贡献1**：首个统一的时空连续7-DoF抓取感知框架
  - 创新点：generation-association 范式——以空间密集预测赋能时间连续跟踪
  - 学术价值：为动态抓取感知建立了新的范式基准
  - 影响范围：机器人抓取、3D视觉、传感器融合

- **贡献2**：将质心感知（COG awareness）引入抓取学习
  - 创新点：受认知科学启发的稳定得分机制
  - 学术价值：连接人类抓取行为研究与机器人抓取

- **贡献3**：真实数据 vs 仿真数据的系统性分析
  - 创新点：用受控实验证明 144 个真实物体 > 数千个仿真物体的训练效果
  - 学术价值：挑战了社区"更多仿真数据"的传统认知

#### 实际应用价值
- **仓储物流清箱**：93.3% 成功率，900+ MPPH，可直接应用于电商仓储
- **动态操作**：抓鱼等动态目标接取，展示了超越传统静态抓取的能力
- **易部署性**：仅需低成本商用深度相机（D415/D435），无需昂贵设备

#### 领域影响
- **短期**：为抓取感知提供了强大的开源基准（已发布代码）
- **中期**：推动社区从仿真数据转向真实数据驱动的范式
- **长期**：建立时空连续感知的统一框架，可扩展到其他机器人操作任务

### 方法优势详解

1. **端到端密集预测**：相比采样-评估方法，一次前向传播覆盖全场景，无需速度-精度权衡
2. **全局上下文感知**：网络以全场景为输入，隐式学习障碍物避免和质心感知
3. **真实数据驱动**：对传感器噪声天然鲁棒，无需复杂 sim-to-real 适配
4. **模块化设计**：时序关联模块在几何处理模块之上独立训练，可灵活组合

### 局限性分析

1. **夹爪配置限制**：仅支持平行夹爪，未扩展到灵巧手或多指手
2. **视觉闭环脆弱**：机器人接近物体时夹爪会遮挡目标，视觉感知失效
3. **对抗物体性能不足**：对极光滑物体（3D打印 PLA 材质）的成功率显著下降
4. **动态跟踪限制**：依赖历史动量预测未来位置，当物体速度突变时失效
5. **无触觉反馈**：缺乏抓取力控制，无法在接触后进行微调
6. **透明/黑色物体**：深度相机无法良好感知，未纳入测试范围

### 适用性与场景分析

**适用场景**：
- 仓储自动化清箱（静态、多类别物体）
- 传送带动态抓取（时间连续跟踪）
- 碎片/薄片精确抓取

**不适用场景**：
- 需要灵巧操作的精细任务 → 需要多指手抓取方法
- 极高反射/透明物体 → 需要额外的 RGB 或偏振感知
- 需要力反馈的装配任务 → 需要结合触觉传感器
- 高动态、高加速度目标 → 需要更高帧率的感知系统

## 与相关论文对比

### 对比论文选择依据
AnyGrasp 建立在 GraspNet-1Billion 和 GSNet 的基础之上，同时与 DexNet 系列构成采样-评估 vs 端到端的范式对比。

### 对比总结

| 对比维度 | GPD/DexNet 4.0 | GraspNet | **AnyGrasp** |
|----------|---------------|----------|--------------|
| 范式 | 采样-评估 | 端到端 | 端到端 + 时序关联 |
| DoF | 6-DoF | 6-DoF | **7-DoF** |
| 场景 | 静态 | 静态 | **静态 + 动态** |
| 推理速度 | 数秒 | ~100ms | **<200ms (含后处理)** |
| 训练数据 | 数千仿真物体 | 100 真实物体 | **144 真实物体** |
| 质心感知 | - | - | **有** |
| 障碍物感知 | 显式碰撞检测 | 隐式 | 隐式 + 显式后处理 |

## 技术路线定位

### 所属技术路线
本文属于**端到端 6/7-DoF 抓取姿态检测**技术路线，核心特点：
- 全局推理：以全场景点云为输入，非局部裁剪
- 密集预测：单次前向传播生成密集覆盖，非稀疏采样
- 端到端学习：从感知直接输出抓取姿态，无需手写特征

### 技术路线发展历程
```
[DexNet 2.0] → [GPD] → [GraspNet-1Billion] → [GSNet] → [AnyGrasp] → [动态+灵巧]
 采样-评估      采样-评估    端到端密集预测      加graspness   时空统一      未来
```

### 本文在技术路线中的位置
- **承上**：直接基于 GSNet 架构，使用 GraspNet-1Billion 数据集
- **启下**：证明了 generation-association 范式的可行性，为动态抓取感知开辟新方向
- **关键节点**：首次将空间密集预测的优势延伸到时间维度

## 未来工作建议

### 作者建议的未来工作
1. 结合视觉和触觉反馈实现闭环抓取调整
2. 将方法扩展到不同类型的机器人手（多指手、吸盘等）

### 基于分析的未来方向
1. **多指手扩展**：将 7-DoF 抓取表示扩展到高维灵巧手配置空间
2. **多模态融合**：结合 RGB 信息处理透明/黑色物体
3. **主动感知**：机器人主动调整视角以解决遮挡问题
4. **强化学习优化**：用 RL 学习抓取后的闭环调整策略

## 我的综合评价

### 价值评分

#### 总体评分
**8.7/10** - 工程扎实、实验全面、具有显著实用价值的工作，在时空统一抓取感知上做出了开创性贡献

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7.5/10 | generation-association 范式新颖，但核心架构基于 GSNet 改进 |
| 技术质量 | 9/10 | 方法严谨，公式清晰，代码开源，后处理设计实用 |
| 实验充分性 | 9.5/10 | 300+物体大规模测试、人类对比、多传感器、动态场景、消融实验极其全面 |
| 写作质量 | 8.5/10 | 结构清晰，逻辑流畅，但部分细节分散在引用论文中 |
| 实用性 | 9/10 | 开源代码、真实硬件验证、可直接部署 |

### 重点关注

- **generation-association 范式**：将空间密集预测与时间跟踪解耦的设计哲学，对其他动态感知任务有借鉴意义
- **真实数据的价值**：用 144 个真实物体击败数千个仿真物体——这对整个机器人学习社区有重要启示
- **全面的大规模评估**：300+ 测试物体的实验规模远超此前所有抓取论文

## 相关论文

### 直接相关
- [[GraspNet-1Billion]] - 本文使用的训练数据集和端到端范式基础
- [[GSNet]] - 本文几何处理模块的基础架构
- [[DexNet 4.0]] - 主要对比基线，采样-评估范式代表

### 后续工作
- 灵巧手抓取扩展
- 结合触觉反馈的闭环抓取

## 外部资源
- [Project Page](https://graspnet.net/anygrasp.html)
- [GraspNet 项目](https://graspnet.net/)
- [YouTube 实验视频合辑](https://youtu.be/dNnLgAGreec)

> [!tip] 关键启示
> 密集空间预测是赋能时间连续跟踪的基石——先让网络"看全"场景，再让网络"记住"抓取，二者结合即可实现时空统一的鲁棒抓取感知。

> [!warning] 注意事项
> - 仅支持平行夹爪，不适用于多指手抓取
> - 训练需要物体 6D 姿态标注（人工标注成本高）
> - 抓鱼实验中近一半失败原因是水中低摩擦导致的滑动，非感知问题
> - 遮挡会导致视觉感知失效，需要额外策略处理

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！机器人抓取领域的里程碑级工作，实验规模空前，工程和学术价值兼具。适合所有从事机器人操作、3D视觉抓取的研究者阅读。
