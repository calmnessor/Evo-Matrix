# Affordance 方向综述：从几何分析到视觉语言引导

> Embodied Affordance — 让机器人理解"此刻在这个场景中，哪里能做什么动作"。本文为科研新手提供该方向的完整学术地图。

## 1. 定义与直觉

**一句话**：Affordance 预测 = 给定场景观测，预测每个空间位置/每个物体**可以执行什么操作**以及**操作的参数**。

**生活化类比**：你走进厨房，扫一眼就能知道——"这个把手可以拉（抽屉），这个平面可以放东西（台面），这个杯柄可以抓（杯子）"。这种"一眼看出物体的使用方式"就是 affordance。机器学习要做的就是把这种人类直觉赋予机器人。

```
传统物体检测:  "这是一个杯子" (What)
Affordance:    "杯柄可以抓，杯口上方可以倒水，杯底可以放稳" (What can I do with it)

更深一层:
  - "可以抓" 还不够 → "用什么样的手势抓？从哪个方向？多大力？"
  - "可以倒水" 还不够 → "倒向哪个方向？倾斜多少度？"
```

### Affordance 概念的学术起源

心理学家 James J. Gibson 在 1977 年提出 "affordance" 概念——环境提供给动物的行动可能性。Gibson 强调 affordance 是**关系的而非主观的**：一个物体能提供什么行动，取决于行动者（agent）的能力和物体的属性之间的关系。

在具身智能语境下，这个概念获得了精确的数学含义：**affordance = 场景状态 s + 动作 a → 成功概率 P(success | s, a)**。

## 2. 为什么重要

| 没有 Affordance 的机器人 | 有 Affordance 的机器人 |
|------------------------|---------------------|
| VLA 直接从像素输出动作（黑盒） | VLA 先理解"哪里可以做动作"再规划 |
| 在新场景中盲目尝试 | 在新场景中"看到"动作可能性 |
| 动作生成无约束 | 动作受 affordance map 约束（更安全、更高效） |
| 抓/推/放 需要分别训练 | 统一的 affordance 表示覆盖所有动作类型 |

**学术定位**：Affordance 是连接**感知**和**行动**的中间表示（intermediate representation）。它比像素更抽象（带有语义），比最终动作更通用（跨任务共享）。在 VLA 架构中，affordance 可作为 V 和 A 之间的桥梁。

## 3. 技术演进时间线

```
传统时代 (2010-2018): 几何为主
├─ 2010以前: 力闭合分析 (Force Closure)，摩擦锥，分析抓取
├─ 2017: Dex-Net 2.0 → 大规模仿真训练抓取质量评估
├─ 2019: Dex-Net 4.0 → 通用抓取的基础 (Science Robotics)

深度学习时代 (2019-2021): 数据驱动崛起
├─ 2019: 6-DOF GraspNet → 端到端 6-DoF 抓取
├─ 2020: GraspNet-1Billion → 10 亿级抓取标注
├─ 2021: Where2Act → 像素级交互 affordance 预测
│        → 预测每个像素"推/拉/抓"的效果
├─ 2021: VAT-MART → 视觉 affordance + 工具使用

表示学习时代 (2022-2023): NeRF/3D 表示
├─ 2022: DexGraspNet → 灵巧手抓取 affordance
├─ 2022: DualAfford → 将 affordance 学习视为"动作可执行性预测"
├─ 2023: IFR-Explore → 主动探索未见的 affordance
├─ 2023: AnyGrasp → 7-DoF 通用抓取 (实时!)
├─ 2023: GAPartNet → 物体部件级 affordance 数据集
├─ 2023: GenAug → 生成式数据增强 → affordance 泛化 (RSS 2023)
├─ 2023: GraspGPT → LLM 常识 + affordance 联合推理 (ICRA 2024)

大模型时代 (2024-2025): VLM + 3DGS 驱动的 affordance
├─ 2024: ManiGaussian → 3DGS + affordance (动态场景, ECCV)
├─ 2024: Splat-MOVER → 3DGS 语义+抓取 affordance + 场景编辑
├─ 2024: GauTOAO → 3DGS 任务导向 affordance，零样本 VLM 引导
├─ 2024: ReKep → 关系关键点约束 → affordance 的结构化表示
├─ 2024: Robo-ABC → 大规模 affordance 数据自动生成 (ECCV)
├─ 2024: FoundationGrasp → LLM+VLM 联合任务导向抓取
├─ 2025: 3DAffordSplat → 首个 3DGS affordance 大规模数据集 + AffordSplatNet (ACM MM Oral)
├─ 2025: SeqAffordSplat → 场景级序列 affordance 推理，LLM 规划 + 3DGS 执行
└─ 2025: π₀.₅+ego → 人类视频 → 机器人 affordance 涌现
```

## 4. 方法分类学

```
Affordance 预测方法
│
├── 路线 A: 几何分析 (Geometric)
│   思路: 3D 重建 → 分析曲面几何 → 判断可操作性
│   代表: Dex-Net, 力闭合分析, GPD
│   优势: 数学严格, 可解释, 不依赖训练数据
│   劣势: 无法理解语义, 泛化到新物体困难
│   适用: 工业场景已知几何的抓取
│
├── 路线 B: 数据驱动的密集预测 (Dense Prediction)
│   思路: RGB-D/点云 → 卷积网络 → 逐像素/逐点 affordance score
│   代表: Where2Act, GraspNet, GG-CNN
│   优势: 快速 (>100fps), 端到端, 数据增强容易
│   劣势: 依赖大规模标注数据, 黑盒泛化难评估
│   适用: 已知物体类型的桌面操作
│
├── 路线 C: 3D 表示学习 (3D Representation)
│   思路: NeRF/3DGS/点云 → 隐式/显式 3D affordance
│   代表: ManiGaussian, GAPartNet, NeuralGrasp
│   优势: 支持新视角, 3D 信息完整, 可处理遮挡
│   劣势: 训练慢, 计算开销大, 对动态场景不稳定
│   适用: 需要精确 3D 信息的精细操作
│   │
│   └── 子路线 C₁: 3DGS Affordance (🔥 2024-2025 爆发)
│       思路: 用 3D Gaussian Splatting 作为 affordance 表征
│       代表: 3DAffordSplat, SeqAffordSplat, GauTOAO, Splat-MOVER
│       优势: 连续密集表征, 实时渲染, 天然适合 affordance 定位
│       关键突破: 从单物体→场景级, 从单步→序列化, 从几何→语义+物理
│
├── 路线 D: VLM/LLM 引导 (Language-Guided)
│   思路: 用 VLM 的常识推理 → 指导 affordance 预测
│   代表: FoundationGrasp, GraspGPT, ManipVQA
│   优势: 零样本泛化, 能理解"这个物体能做什么"的常识
│   劣势: VLM 昂贵, 对没见过物体可能出错, 容易"脑补"
│   适用: 开放世界未见物体、语言交互场景
│
└── 路线 E: 交互探索 (Interactive Exploration)
    思路: 主动与环境交互 → 从反馈中学习 affordance
    代表: IFR-Explore, Curious Robot, Robo-ABC
    优势: 可以发现数据集标注中没有的 affordance
    劣势: 探索成本高 (需要真实/仿真交互), 安全性挑战
    适用: 新环境探索, 需要发现"意外可用性"的场景
```

## 5. 代表工作深度对比

| 维度 | Where2Act | AnyGrasp | ManiGaussian | 3DAffordSplat | SeqAffordSplat | GraspGPT | ReKep |
|------|-----------|----------|-------------|----------------|-----------------|----------|-------|
| **年份** | 2021 | 2023 | 2024 | 2025 | 2025 | 2023 | 2024 |
| **输入** | 部分点云 | 点云 | 多视角 RGB-D | 3DGS + 点云 | 3DGS + 语言指令 | RGB + 文本 | RGB-D |
| **输出** | 逐点 (推/拉/抓) | 6-DoF 抓取位姿 | 3D affordance + 操作轨迹 | 3D affordance 分割 | 序列化 3D affordance masks | 抓取位姿 + 文本解释 | 关系关键点约束 |
| **动作类型** | 推/拉/抓 | 抓取 | 抓/推/按/开 | 18 种 affordance | 18 种 affordance (多步) | 抓取 | 任意 (约束定义) |
| **泛化** | 同类别物体 | 开放世界物体 | 动态场景 | 单物体→场景 | 场景级多物体序列 | 语言引导零样本 | 结构化泛化 |
| **推理速度** | 快 | 实时 (>30fps) | 慢 (3DGS) | 中等 | 中等 (含 LLM 规划) | 中等 | 快 |
| **开源性** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **核心创新** | 像素级 affordance | 7-DoF 通用 | 3DGS 动态 | 首个 3DGS affordance 数据集 | 序列化推理 + LLM 规划 | LLM 常识 | 约束数学化 |

## 6. 当前 SOTA 与趋势 (2025)

### 已解决的问题
- ✅ 已知物体的 6-DoF 抓取位姿预测 → AnyGrasp 已达实用水平
- ✅ 简单场景的像素级 affordance 分割 → 准确率 >85%
- ✅ 桌面场景 "可抓/不可抓" 的二分类 → 近乎解决

### 正在突破的方向
- 🔥 **3DGS affordance**（3DAffordSplat / SeqAffordSplat / GauTOAO）：从点云切换到 3DGS 表征，连续密集的高斯分布天然适合精细 affordance 定位
- 🔥 **动态 affordance**（ManiGaussian）：物体状态会变 → affordance 也要动态更新
- 🔥 **序列化 affordance**（SeqAffordSplat）：不再是单步"能做什么"，而是多步"先做什么再做什么"，LLM 负责规划、3DGS 负责定位
- 🔥 **语言引导 affordance**（GraspGPT / FoundationGrasp）："抓最左边的杯子" → 语言+affordance 联合
- 🔥 **大模型 scaling**（Robo-ABC）：用 VLM 自动生成 affordance 标注数据
- 🔥 **Relation affordance**（ReKep）：不只预测"能抓"，还预测"抓哪里才能完成后续任务"
- 🔥 **人机迁移**（π₀.₅+ego）：人类视频中自动提取 affordance 知识

### 四大趋势
1. **从几何到语义**：不再只看几何可抓性，还要理解物体功能和任务上下文
2. **从点云到 3DGS**：3D Gaussian Splatting 正快速取代点云成为 affordance 的默认 3D 表征——连续、密集、可微分、实时渲染，2024-2025 年该方向论文爆发式增长
3. **从单步到序列**：affordance 不再只看"下一步"，SeqAffordSplat 等开始做长周期多步推理
4. **从单点到关系**：多个物体之间的 affordance 关系（"锤子可以用来敲这个钉子" → 跨物体 affordance）

## 7. 关键挑战与开放问题

| 挑战 | 现状 | 可能突破方向 |
|------|------|------------|
| **功能 affordance 理解** | 模型知道"杯柄可以抓"，但不知道"杯子用来喝水" | VLM 常识注入, 功能知识图谱 |
| **动态 affordance** | 大多数方法假设场景静态 | 3DGS 动态更新, 实时 SLAM + affordance |
| **序列化 affordance** | 当前只看"下一步能做什么"，SeqAffordSplat 刚起步 | LLM 任务规划 + 3D affordance 联合推理 |
| **跨物体 affordance** | 单物体 affordance 成熟，多物体联合 affordance 极少 | 图神经网络 (物体关系图), ReKep 约束推理 |
| **灵巧手 affordance** | 平行夹爪的 affordance 成熟，五指灵巧手极少 | 接触点优化, 强化学习探索 |
| **3DGS→动作闭环** | 3DGS 能做 affordance 定位，但到动作生成还有 gap | 3DGS affordance → VLA 动作空间映射 |
| **触觉 affordance** | 几乎只有视觉，缺乏触觉 affordance | 触觉传感器普及 + 多模态融合 |
| **评估标准** | 缺乏统一的 affordance benchmark | 需要类似 LIBERO 的 affordance 专用 benchmark |

## 8. 与其它方向的关系

```
Affordance 在具身智能技术栈中的位置:

输入层:     Camera → 图像/点云
               ↓
感知层:     [物体检测] → 这是什么
               ↓
Affordance: [可供性预测] → 哪里能做什么 ← 我们在这里!
               ↓
决策层:     VLA → 选择哪个 affordance + 生成动作
               ↓
执行层:     [Grasp] → 具体抓取执行
           [Motion] → 运动规划
               ↓
反馈层:     世界模型 → 预测动作后果 → 更新 affordance 理解
```

- **Affordance → VLA**: affordance map 条件化 VLA 的动作空间，减少搜索范围 → 动作更安全、更高效
- **Affordance ⊃ Grasp**: 抓取是 affordance 测 "抓" 这个动作类型。Grasp 社区的方法很多可直接用于 affordance
- **Affordance ← 世界模型**: 世界模型预测 "做动作 X 后会怎样" → 自动标注 affordance
- **Affordance ← 仿真**: 在仿真中大规模探索 affordance → 转移到真机

## 9. 必读论文清单

### 入门级
| 论文 | 年份 | 一句话 | 阅读时间 |
|------|------|--------|---------|
| **Where2Act** | 2021 | 像素级 affordance 的开创性工作 | 2h |
| **Dex-Net 4.0** | 2019 | 大规模分析抓取的经典 (Science Robotics) | 1.5h |
| **AnyGrasp** | 2023 | 7-DoF 通用抓取，实时性极好 | 1.5h |

### 进阶级
| 论文 | 年份 | 一句话 | 阅读时间 |
|------|------|--------|---------|
| **ManiGaussian** | 2024 | 3DGS + affordance → 动态场景 | 2.5h |
| **3DAffordSplat** | 2025 | 首个 3DGS affordance 大规模数据集 + AffordSplatNet (ACM MM Oral) | 2h |
| **GAPartNet** | 2023 | 物体部件级 affordance 数据集 | 2h |
| **ReKep** | 2024 | 关系关键点 → structured affordance | 2h |

### 前沿级
| 论文 | 年份 | 一句话 | 阅读时间 |
|------|------|--------|---------|
| **SeqAffordSplat** | 2025 | 场景级序列 affordance 推理，LLM 规划 + 3DGS 执行 | 3h |
| **GraspGPT** | 2023 | LLM 常识 + affordance 联合推理 (ICRA 2024) | 2.5h |
| **Robo-ABC** | 2024 | 大规模 affordance 数据自动生成 (ECCV) | 3h |
| **π₀.₅+ego** | 2025 | 人类视频 → affordance 涌现 | 2h |

## 10. 新手学习路径

### 第 1 周：建立直觉
- Day 1-2: 读 Gibson 原始 affordance 概念 + Where2Act 论文
- Day 3-4: 跑通 AnyGrasp demo（在自己的点云上测试）
- Day 5-7: 理解 affordance 的表征形式（稀疏/密集/参数化）

### 第 2 周：深入方法
- Day 1-3: 读 GAPartNet + ManiGaussian，理解 3D affordance 表示
- Day 4-5: 对比几何方法和学习方法的核心差异；读 3DAffordSplat，理解 3DGS 为什么适合 affordance
- Day 6-7: 在 robosuite 中运行 affordance 预测 + 抓取 pipeline

### 第 3 周：跟上大模型时代
- Day 1-3: 读 GraspGPT/FoundationGrasp，理解语言与 affordance 的结合
- Day 4-5: 读 ReKep，理解关系型 affordance
- Day 6-7: 思考：你的课题中 affordance 能解决什么瓶颈？

### 第 4 周：动手 + 思考
- Day 1-3: 尝试用 CLIP/SigLIP 做零样本 affordance 分类
- Day 4-5: 在自己的数据上训练一个简单的 affordance 预测器
- Day 6-7: 写 mini-review：affordance 领域未来 3 年的最重要方向

## 11. 资源汇总

### 开源代码
| 代码库 | 内容 | 链接 |
|--------|------|------|
| **AnyGrasp** | 通用 7-DoF 抓取 | github.com/graspnet/anygrasp_sdk |
| **Where2Act** | 像素级 affordance | github.com/daerduo/Where2Act |
| **GAPartNet** | 部件级 affordance | gapartnet.github.io |
| **ManiGaussian** | 3DGS affordance | github.com/ManiGaussian |
| **3DAffordSplat** | 首个 3DGS affordance 数据集+模型 (ACM MM 2025) | github.com/HCPLab-SYSU/3DAffordSplat |
| **SeqAffordSplat** | 场景级序列 affordance 推理 | (arXiv: 2507.23772) |
| **ReKep** | 关系关键点 | rekep-robot.github.io |

### 数据集
| 数据集 | 内容 | 规模 |
|--------|------|------|
| PartNet-Mobility | 关节物体 3D 模型 | 2000+ 实例 |
| GAPartNet | 部件级 affordance 标注 | 8000+ 物体 |
| GraspNet-1B | 抓取位姿标注 | 10 亿级 |
| ACD (Affordance-Centric Dataset) | 以 affordance 为中心 | — |

### 教程与资源
| 资源 | 类型 | 链接 |
|------|------|------|
| Gibson 原著 | 理论 | "The Ecological Approach to Visual Perception" |
| CS 285 | 课程 | 含 affordance 相关讲座 |
| 具身智能前沿 | 社区 | 关注 affordance 专刊 |

## 关联笔记

- [[📌-Affordance总览]] — Affordance 方向快速导航
- [[../02-VLA/VLA方向综述|VLA 综述]] — affordance 的下游消费者
- [[../04-Grasp/Grasp方向综述|Grasp 综述]] — affordance 中"抓取"子集的专项
- [[../05-世界模型/世界模型方向综述|世界模型 综述]] — 提供 affordance 的反馈信号
