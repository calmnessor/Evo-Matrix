# 📚 第二阶段: 30 天深度学习计划

> **前提**: 已完成为期两周的速通阶段  
> **目标**: 从"会用"到"能改"到"能造"  
> **时间**: 每天 8 小时，共 30 天

---

## 🗺️ 第二阶段路线图

```
速通阶段（已完成）
    │
    ▼
第二阶段: 深入每个模块
    │
    ├── 第 3 周: 3D 视觉与空间理解
    ├── 第 4 周: 强化学习与机器人控制
    ├── 第 5 周: 扩散模型深入
    └── 第 6 周: 多模态架构 + 开源贡献
    │
    ▼
实战项目: 3D-VLA / 扩散对比 / 开源贡献
    │
    ▼
达到实习/初级研究员水平 🎯
```

---

## 第 3 周: 3D 视觉与空间理解

### 为什么重要？
VLA 中的"Vision"不仅仅是 2D 图像。真实机器人需要理解 3D 空间——物体的位置、深度、姿态。目前的 VLA 大多只用 2D 视觉，加入 3D 是明显的改进方向。

### Day 15-16: 3D 视觉基础
- 点云表示 (Point Cloud)
- 深度图 (Depth Map)
- 相机模型: 针孔相机、内参矩阵 K、外参 [R|t]
- 手眼标定 (Hand-Eye Calibration)

### Day 17-18: PointNet 系列
- PointNet: 首个点云深度学习网络
- PointNet++: 层次化点云特征学习
- 代码实战: 用 PointNet++ 做物体分类

### Day 19-20: 3D 重建
- NeRF (Neural Radiance Fields) 基础
- Gaussian Splatting 概览
- 在 VLA 中的应用: 从 2D 图像构建 3D 场景表示

### Day 21: VLA + 3D
- 阅读 3D-VLA 相关论文
- 理解 3D 特征如何融入现有 VLA pipeline
- 设计一个简单的 3D-VLA 实验

---

## 第 4 周: 强化学习与机器人控制

### Day 22-23: RL 基础回顾
- MDP (Markov Decision Process)
- 策略梯度 (Policy Gradient)
- Actor-Critic 架构

### Day 24-25: PPO 深入
- PPO (Proximal Policy Optimization)
- GAE (Generalized Advantage Estimation)
- 代码: 用 PPO 训练 MuJoCo 环境

### Day 26-27: RL 与 VLA 的结合
- 在线 RL 微调 VLA
- RLHF 在机器人中的应用
- 从 Human Preference 学习

### Day 28: Sim-to-Real
- Domain Randomization 原理与实践
- System Identification
- 仿真域适应到真实机器人

---

## 第 5 周: 扩散模型深入

### Day 29-30: DDPM 数学推导
- 从 ELBO 推导训练损失
- 正向扩散的马尔可夫性质
- 反向过程的参数化

### Day 31-32: 加速采样
- DDIM (Denoising Diffusion Implicit Models)
- 减少采样步数的方法
- 代码: 对比 DDPM vs DDIM 采样质量

### Day 33-34: Diffusion Policy 进阶
- CNN-based vs Transformer-based 对比
- 条件注入方式: FiLM, Cross-Attention, AdaLN
- Receding Horizon Control 与扩散策略

### Day 35: Flow Matching
- 连续归一化流 (Continuous Normalizing Flows)
- Flow Matching 与扩散模型的关系
- π₀ 的 Flow Matching 实现

---

## 第 6 周: 多模态架构 + 开源贡献

### Day 36-37: Cross-Attention 深入
- Encoder-Decoder Attention 模式
- 不同模态间的信息流动
- Q-Former (BLIP-2) 架构

### Day 38-39: 多模态融合方式
- Early Fusion (输入层拼接)
- Middle Fusion (中间层交叉注意力)
- Late Fusion (输出层融合)
- 各种方式的优劣对比

### Day 40-41: 论文精读
- Octo 论文完整精读
- π₀ 论文完整精读
- GR00T 论文略读

### Day 42: 开源贡献
- 给 OpenVLA 提 PR（bug fix / 文档 / 小功能）
- 写一篇 VLA 入门博客/笔记
- 在 GitHub 上展示自己的项目

---

## 🎯 第二阶段毕业标准

完成以下至少 5 项:

- [ ] 用 PointNet++ 完成一个 3D 物体分类任务
- [ ] 用 PPO 训练一个 MuJoCo 机器人控制策略
- [ ] 实现 DDIM 加速采样并可视化对比
- [ ] 写一篇 Diffusion Policy vs BC 的技术对比笔记
- [ ] 给 OpenVLA 提交一个 PR
- [ ] 在仿真环境中部署自己的 VLA 策略
- [ ] 精读并复现一篇 VLA 论文的核心方法

---

## 📖 第二阶段核心论文清单

| 论文 | 精读/略读 | 预计时间 |
|------|-----------|----------|
| PointNet | 精读 | 2h |
| PointNet++ | 精读 | 2h |
| NeRF | 略读 | 1h |
| PPO (原始论文) | 精读 | 2h |
| DDPM | 精读 | 3h |
| DDIM | 精读 | 2h |
| Diffusion Policy | 精读 | 3h |
| Flow Matching | 精读 | 2h |
| Octo | 精读 | 3h |
| π₀ | 精读 | 3h |
| GR00T | 略读 | 1h |

---

> 📅 **预计完成日期**: 2026 年 6 月 20 日  
> 每天记录进度，遇到问题及时查漏补缺！
