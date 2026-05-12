---
date: "2019-12-31"
paper_id: "arXiv:1912.13470"
title: "GraspNet: A Large-Scale Clustered and Densely Annotated Dataset for Object Grasping"
authors: "Hao-Shu Fang, Chenxi Wang, Minghao Gou, Cewu Lu"
domain: "Grasp"
tags:
  - 论文笔记
  - Grasp
  - Dataset
  - 6-DoF
  - Force-Closure
  - Benchmark
  - Point-Cloud
quality_score: "9.0/10"
created: "2026-05-12"
updated: "2026-05-12"
status: analyzed
---

# GraspNet (GraspNet-1Billion)

> **注意**：该论文最早以 "GraspNet: A Large-Scale Clustered and Densely Annotated Dataset for Object Grasping" 发布在 arXiv (1912.13470, 2019-12-31)。后续 CVPR 2020 版本更名为 **GraspNet-1Billion**，将数据扩展为 97,280 张图像 + 12 亿抓取姿态。

## 核心信息
- **论文ID**：arXiv:1912.13470
- **作者**：Hao-Shu Fang（方浩树）, Chenxi Wang（王晨曦）, Minghao Gou（苟铭浩）, Cewu Lu（卢策吾）
- **机构**：上海交通大学 MVIG 实验室
- **发布时间**：2019-12-31（arXiv v1）/ CVPR 2020（正式发表）
- **会议/期刊**：CVPR 2020
- **链接**：[arXiv](https://arxiv.org/abs/1912.13470) | [Project Page](https://graspnet.net/)
- **引用**：[GraspNet-1Billion@CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html)

## 摘要翻译

### 英文摘要
Object grasping is critical for many applications, which is also a challenging computer vision problem. However, for the clustered scene, current researches suffer from the problems of insufficient training data and the lacking of evaluation benchmarks. In this work, we contribute a large-scale grasp pose detection dataset with a unified evaluation system. Our dataset contains 87,040 RGBD images with over 370 million grasp poses. Meanwhile, our evaluation system directly reports whether a grasping is successful or not by analytic computation, which is able to evaluate any kind of grasp poses without exhausted labeling pose ground-truth. We conduct extensive experiments to show that our dataset and evaluation system can align well with real-world experiments.

### 中文翻译
物体抓取对许多应用至关重要，同时也是一个具有挑战性的计算机视觉问题。然而，对于杂乱场景，当前研究面临训练数据不足和评估基准缺乏的问题。本工作贡献了一个大规模抓取姿态检测数据集及统一评估系统。数据集包含 87,040 张 RGB-D 图像，超过 3.7 亿个抓取姿态。同时，评估系统通过分析计算直接报告抓取是否成功，无需耗尽的标注真值即可评估任意类型的抓取姿态。大量实验表明，该数据集和评估系统能够与真实世界实验良好对齐。

### 核心要点提炼
- **研究背景**：杂乱场景的 6-DoF 抓取检测缺乏大规模训练数据和统一评估基准
- **研究动机**：现有抓取数据集要么规模小（人工标注）、要么是模拟数据（存在视觉域间隙），需要一个"真实图像 + 密集标注"的规模化数据集
- **核心方法**：真实世界采集图像 → 首帧标注物体 6D 姿态 → 相机姿态传播 → 基于力封闭（force closure）分析计算的自动抓取标注 → 在线评估系统
- **主要结果**：88 个物体的高精度 3D 模型 + 170 个杂乱场景 + 87K RGB-D 图 + 3.7 亿抓取姿态；真实机器人验证高分抓取成功率达 96%
- **研究意义**：抓取检测领域的事实标准基准，启发了后续大量 6-DoF 抓取工作

## 研究背景与动机

### 领域现状（2019 年）
抓取检测算法已有诸多进展，但存在两大核心障碍：

1. **表示与评估不统一**：矩形表示（2D）和 6-DoF 表示各有不同评估指标，方法间无法公平对比。真机评估成本高且不可复现。
2. **大规模高质量训练数据缺失**：
   - 人工标注（Cornell, VMRD）→ 规模小、标注稀疏
   - 模拟环境（Dex-Net, Jacquard）→ 规模大，但 visual domain gap 降低真实场景性能

### 研究动机
需要一个满足三个条件的数据集：
- (a) 与真实传感器视觉感知对齐
- (b) 密集、准确的大规模抓取姿态真值标注
- (c) 能统一评估不同抓取表示

核心洞察：**真实世界采集 + 模拟器中分析计算标注 = 兼具真实视觉和规模化标注**

## 研究问题

### 核心研究问题
如何构建一个大规模、真实视觉感知、密集 6-DoF 抓取标注、且具有统一评估系统的抓取数据集？

## 方法概述

### 核心思想
"真实采集 + 计算标注"：用真机实拍获取真实 RGB-D 图像，再借助高精度物体 3D 模型 + 力封闭分析，在模拟器中自动为每帧生成数千个 6-DoF 抓取姿态标注。

### 方法框架

![[overview_of_work.pdf|800]]

> 图1：GraspNet 构建方法总览。(a) 真实世界数据采集；(b) 物体 3D 模型获取；(c) 两阶段自动标注——单物体抓取采样 + 场景投影；(d) 统一评估系统

#### 阶段 1：数据采集（Data Collection）

- **物体选择**：88 个日常物体（32 YCB + 13 DexNet adversarial + 43 自行收集），形状/纹理/大小/材质各异
- **采集设备**：两台 RGB-D 相机（Intel RealSense 435 + Kinect v4 Azure）附着在机械臂上同时拍摄
- **场景构建**：每次随机选 ~10 个物体，杂乱摆放于桌面上
- **采集轨迹**：机械臂沿固定轨迹移动，覆盖 1/4 球面的 256 个不同视角
- **输出**：每帧保存两相机的同步 RGB-D 图 + 相机姿态，共 170 个场景 × 512 帧 = 87,040 张图像

> 关键设计：记录相机姿态使得物体 6D 姿态只需标注首帧，大幅降低标注成本

#### 阶段 2：数据标注（Data Annotation）

**2A. 物体 6D 姿态标注**

只需要标注每个场景第一帧的物体 6D 姿态，后续帧通过相机姿态自动传播：

$$\mathbf{P}_i^j = \mathbf{cam}_i^{-1}\mathbf{cam}_0\mathbf{P}_0^j$$

其中 $\mathbf{P}_i^j$ 为物体 $j$ 在第 $i$ 帧的 6D 姿态，$\mathbf{cam}_i$ 为第 $i$ 帧的相机姿态。再用 ICP 精炼 + 人工检查。

**2B. 抓取姿态自动标注（两阶段）**

![[grasp_annotation_1.pdf|800]]

> 图2：两阶段自动抓取标注流程

**第一阶段：单物体抓取采样**

对每个物体的高精度 3D 网格模型：
1. 降采样得到均匀分布的抓取点（voxel space）
2. 每个抓取点采样 $V$ 个球面均匀分布的接近方向
3. 在 $D \times A$ 的二维网格中搜索抓取候选（$D$ = 抓取深度集合，$A$ = 面内旋转角集合）
4. 夹爪宽度自动适配物体，避免空抓和碰撞

**抓取质量评分（Force Closure Metric）**：

以 $\Delta\mu = 0.1$ 为间隔，从 0.1 递增摩擦系数 $\mu$，直到力封闭条件成立。评分 $s$ 定义为：

$$s = 1.1 - \mu$$

- $s$ 越大 → 所需 $\mu$ 越小 → 更容易抓取
- $\mu = 0.1$ 时成立的抓取 $s = 1.0$（最优）
- $\mu = 1.0$ 时成立的抓取 $s = 0.1$（勉强）

本质：**需要的摩擦力越小，抓取越稳定**。

**第二阶段：场景投影**

将单物体抓取姿态通过物体 6D 姿态投影到世界坐标系：

$$\mathbb{G}^i_{(w)} = \mathbf{P}^i \cdot \mathbb{G}^i_{(o)}$$

$\mathbf{P}^i = \mathbf{cam}_0\mathbf{P}_0^i$ 是物体 $i$ 的世界位姿。投影后做碰撞检测剔除无效抓取。

**标注统计**：
- 每个场景 1.5M ~ 2.5M 个抓取姿态
- 总计 **3.7 亿+** 抓取姿态
- 正负样本比例约 **1:2**

#### 阶段 3：统一评估系统

**传统指标的缺陷**：
- 只能评估矩形表示
- 容差过高（30°旋转 + 0.25 IoU）
- Cornell 数据集准确率已达到 99%，无区分度

**GraspNet 的在线评估**：

**步骤 1：单抓取评估**
- 对每个预测抓取 $\hat{\mathbf{P}}_i$，找出夹爪内的点云以关联目标物体
- 使用力封闭度量在不同 $\mu$ 下计算二值标签

**步骤 2：Pose-NMS**
- 抓取距离定义为元组：$D(\mathbf{G}_1, \mathbf{G}_2) = (d_t, d_{\alpha})$
- $d_t = \|\mathbf{t}_1 - \mathbf{t}_2\|$（平移距离）
- $d_{\alpha} = \arccos\frac{1}{2}(\text{tr}(R_1 R_2^T) - 1)$（旋转距离）
- 阈值：$th_d = 1\text{cm}, th_{\alpha}=5^\circ$

**步骤 3：Precision@k & AP**
- 取每物体 top-$K=10$ 抓取（按置信度排序）
- $AP$ = 对 $k \in [1, 50]$ 的 Precision@k 取平均
- $AP_\mu$ 在不同 $\mu$ 下计算，$AP$ 从 $\mu=0.1$ 到 $\mu=0.5$ 取平均

**数据集划分**：

| 划分 | 场景数 | 描述 |
|------|--------|------|
| Train | 100 | 训练场景 |
| Test-Seen | 30 | 见过但不同摆放的物体 |
| Test-Similar | 30 | 未见过但相似的物体 |
| Test-Novel | 10 | 全新物体 |
| **Total** | **170** | -- |

### 数据集组成

![[component_of_dataset.pdf|800]]

> 图3：GraspNet 数据集关键组成：RGB-D 图 + 物体 6D 姿态 + 抓取姿态 + 物体掩码 + 包围盒 + 相机姿态

## 实验结果

### 真实机器人验证

#### 物体 6D 姿态投影

![[object_6d_pose.png|400]]

> 图4：通过 ArUco 码获取物体 6D 姿态，将抓取姿态投影到相机坐标系进行真机执行

#### 抓取成功率 vs 抓取评分

| 抓取评分 $s$ | 平均成功率 | 代表意义 |
|-------------|-----------|----------|
| $s \geq 0.9$ | **96%** | 极低摩擦要求，几乎总是成功 |
| $s = 0.1$ | 很低 | 勉强力封闭，需要高摩擦 |

> 抓取姿态评分与真机成功率高度一致，验证了力封闭评分的有效性

![[grasping_poses.png|400]]

> 图5：真实机器人抓取执行示例

## 深度分析

### 研究价值评估

#### 理论贡献
- **"真实采集 + 计算标注"范式**：开创性地提出用分析计算替代人工标注来生成密集 6-DoF 抓取真值，成为后续大规模抓取数据集的标准范式
- **力封闭在线评估系统**：不需要预计算真值，可评估任意抓取表示（矩形/6-DoF/任何未来表示），真正统一了评估标准
- **抓取质量评分与摩擦系数的显式关联**：$s = 1.1 - \mu$ 的设计将物理约束量化为可比的分数

#### 实际应用价值
- 成为 **6-DoF 抓取检测的事实标准基准**（到 2025 年几乎所有 6-DoF 抓取论文都在 GraspNet 上评测）
- 88 个物体的高精度模型被后续大量工作复用
- API 和工具链的发布促进了社区标准化

#### 领域影响
- **短期**（2020-2021）：立即成为抓取检测领域最大、最权威的基准
- **中期**（2022-2023）：衍生出 GraspNet-1Billion (CVPR 2020)、AnyGrasp (CVPR 2023) 等后续工作
- **长期**（2024+）：奠定了 Data-Centric Robotics 的思维，影响了后续数据集（如 Grasp-Anything）的设计理念

### 方法优势详解

#### 优势 1：极大规模的密集标注
- 3.7 亿个抓取姿态 vs 之前最大的 Dex-Net（670 万个），两个数量级的领先
- 每个物体 ~420 万个抓取姿态，覆盖连续抓取空间的各个方向

#### 优势 2：真实的视觉感知
- 使用真实相机而非模拟器渲染，消除了视觉 domain gap
- 两种不同相机确保算法对深度质量的鲁棒性

#### 优势 3：表示无关的评估系统
- 不需预计算真值 → 可评估任意抓取表示
- 多种 $\mu$ 下的 AP 提供了更细粒度的性能剖面
- Seen/Similar/Novel 划分科学评估泛化能力

#### 优势 4：低标注成本的可扩展性
- 首帧标注 + 相机传播 → 87K 帧的 6D 姿态标注成本降低到原始方案的 ~1/256
- 物体模型可重复使用（one-time cost）

### 局限性分析

#### 局限 1：已知物体的假设
- 需要预先获得物体的高精度 3D 网格模型
- 对新物体的泛化受限于能否获得 CAD 模型
- 后续工作 AnyGrasp 尝试解决此问题

#### 局限 2：仅平行板夹爪
- 力封闭分析基于两指平行板夹爪，不支持多指手和吸盘
- 夹爪类型的假定限制了数据集的应用范围

#### 局限 3：桌面场景为主
- 采集场景均在桌面上，视角轨迹覆盖 1/4 球面而非完整球面
- 对于 shelf-picking 等特殊场景的代表性有限

#### 局限 4：静态场景假设
- 所有数据为静态场景，不支持动态抓取和运动物体
- 物体仅处于静止摆放状态，不包括滚动/滑动等动态情况

### 适用性与场景分析

#### 适用场景
- **6-DoF 抓取算法研究与评测**：事实标准基准
- **杂乱场景抓取**：多物体多抓取的密集标注场景
- **领域泛化研究**：Seen/Similar/Novel 划分适合测试跨物体泛化
- **物体 6D 姿态估计**：附带的高质量 6D 姿态标注也常用于姿态估计

#### 不适用场景
- **未知/新物体的零样本抓取**（需要 CAD 模型）
- **柔性/可变形物体**（力封闭分析基于刚性假设）
- **非平行板夹爪**（多指手、吸盘等）
- **移动操作 / 动态抓取**

## 与相关论文对比

| 对比维度 | Cornell | Jacquard | Dex-Net 2.0 | **GraspNet（本文）** |
|----------|---------|----------|-------------|---------------------|
| 数据来源 | 真实 | 模拟 | 模拟 | **真实图像 + 模拟标注** |
| 物体数 | 240 | 11K | 1,500 | **88（高质量模型）** |
| 场景数 | 单物体 | 单物体 | 孤立物体 | **170 杂乱场景** |
| 样本数 | 1035 | 54K | 6.7M | **87K（CVPR版97K）** |
| 抓取表示 | 矩形 | 矩形 | 矩形+质量分 | **6-DoF + 矩形** |
| 标注量 | 8019 抓取 | 1.1M 抓取 | 6.7M 抓取 | **3.7亿抓取（CVPR版12亿）** |
| 评估指标 | 矩形 IoU | 矩形 IoU | 物理抓取质量 | **力封闭在线评估 + AP** |
| 年份 | 2011 | 2018 | 2017 | **2019/CVPR2020** |

**关键区别**：GraspNet 是第一个同时满足"真实视觉 + 密集 6-DoF 标注 + 统一评估"的数据集，其 "real-images-with-synthetic-labels" 策略成为新范式。

## 技术路线定位

### 所属技术路线
本文属于 **Real-to-Sim-to-Real Grasping Benchmark** 路线：

```
Cornell (2011, 矩形)
  → Dex-Net (2017, 模拟+物理分析)
    → Jacquard (2018, 模拟大规觧)
      → [本文] GraspNet (2020, 真实图像+分析标注)
        → GraspNet-1Billion (2020, 扩展版)
          → AnyGrasp (2023, 任意物体)
```

### 本文在技术路线中的位置
- **承上**：继承了 Dex-Net 的力封闭分析方法，借鉴了 Cornell 的矩形表示，吸收了 Jacquard 的规模化思维
- **启下**：开启了"真实图像 + 模拟标注"范式，为 AnyGrasp、Grasp-Anything 等后续工作提供了数据基础和评估框架
- **关键节点**：从"小数据集 + 各自评测"到"大数据集 + 统一评测"的转折点

## 未来工作建议

### 作者建议的扩展方向
1. **多指手和吸盘抓取**：将标注扩展到更多夹爪类型
2. **更多物体的 3D 模型**：扩大物体种类覆盖面

### 基于分析的未来方向
1. **CAD-free 抓取标注**：利用 NeRF/3DGS 替代 CAD 模型，使标注流程不依赖物体先验
2. **动态/交互抓取**：增加物体移动/滚动等动态场景的数据
3. **全视角球面覆盖**：将相机轨迹扩展到完整球面，支持任意方向的抓取

## 我的综合评价

### 价值评分

#### 总体评分
**9.0/10** — CVPR 2020，抓取检测领域里程碑。范式创新（真实采集+分析标注）、数据规模碾压、评估系统科学。几乎所有的后续 6-DoF 抓取工作都在该数据集上评测，已成为事实标准。唯一的扣分项是依赖已知 CAD 模型和仅支持平行板夹爪。

#### 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | "真实采集+计算标注"是范式级创新，在线力封闭评估系统是评估方面的原创贡献 |
| 技术质量 | 9/10 | 两阶段标注流程设计精妙，力封闭评分与物理直觉高度吻合，相机姿态传播大幅降低标注成本 |
| 实验充分性 | 8/10 | 真机实验验证了标注质量（96%成功率），但未提供与其他数据集的"在此训练→彼测试"交叉实验 |
| 写作质量 | 8/10 | 结构清晰，图表到位，但一些关键细节（如 NMS 具体算法）放在了补充材料 |
| 实用性 | 10/10 | 成为领域标准基准，API/工具链完善，社区广泛采用——实用性的天花板 |

### 重点关注

- **力封闭评估的无真值特性**：这意味着测试集的"答案"实际上在评估时才被计算出来，而非预先标注。这种设计使得任何形式的抓取表示都可以被评估。
- **$s = 1.1 - \mu$**：这个简单的评分公式巧妙地将物理约束（摩擦系数）映射到了可比较的分数空间。
- **Seen/Similar/Novel 划分**：科学的三级泛化评估，比简单的 train/test split 更能揭示方法的真实泛化水平。

## 我的笔记

%% GraspNet 可能是近十年来影响最大的抓取数据集之一。它的核心方法论洞察——"用真实相机获取视觉信号，用模拟器做物理分析标注"——远比具体的 88 个物体或 170 个场景更有价值。这个思路后来被广泛采用。

对于做 VLA 和具身智能的研究者来说，理解 GraspNet 的评估系统非常重要。它的 AP 指标考虑了多个摩擦系数水平，这意味着模型不仅要预测"能抓到"的姿态，还要预测"在低摩擦下也能抓到"的鲁棒姿态。

CVPR 版本（GraspNet-1Billion）将此 arXiv 版本扩展为 97K 图像 + 12 亿抓取姿态，并增加了端到端抓取检测网络。引用时应该引用 CVPR 版本。 %%

## 相关论文

### 直接相关
- [[../Grasp-Anything/Grasp-Anything|Grasp-Anything]] (ICRA 2024) — 继承了"规模化抓取数据集"思路，但改用基础模型合成数据

### 背景相关
- Dex-Net 2.0 (Mahler et al. 2017) — 力封闭分析方法的先驱
- Jacquard (Depierre et al. 2018) — 大规模模拟抓取数据集
- Cornell (Jiang et al. 2011) — 最早的抓取检测数据集

### 后续工作
- AnyGrasp (Fang et al. CVPR 2023) — 同一团队，扩展到任意物体
- GraspNet-1Billion (CVPR 2020) — 本文的正式扩展版

## 外部资源
- 项目主页：[https://graspnet.net/](https://graspnet.net/)
- API 仓库：[https://github.com/graspnet/graspnetAPI](https://github.com/graspnet/graspnetAPI)
- CVPR 2020 版：[https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion)

> [!tip] 关键启示
> "真实采集 + 模拟器分析标注"的混合策略，同时解决了视觉域间隙和标注成本问题，成为大规模机器人数据集的范式模板。

> [!warning] 注意事项
> - 标注需要物体的高精度 CAD 模型——对于没有模型的物体需要先扫描重建
> - 力封闭评估仅适用于刚性物体 + 平行板夹爪
> - 引用时应优先引用 CVPR 2020 版本（GraspNet-1Billion）而非此 arXiv 版本

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！如果你是做抓取检测或机器人操作的研究者，这是必读的领域基石论文。
