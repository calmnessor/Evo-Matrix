# Affordance 预测研究方向

> Embodied Affordance Prediction — 理解物体"能用来做什么"

## 方向简介

Affordance 预测是具身智能的核心感知能力之一：给定场景观测，预测物体表面/空间哪些区域可以执行特定动作（如抓取、推动、打开、放置），以及动作的参数（抓取点、方向、力度）。

### 什么是 Affordance？

```
传统视觉: 这是什么？ → "杯子"
Affordance: 能怎么用？ → "杯柄处可以抓", "杯口上方可以倒水"
```

## 核心概念

### Affordance 表征形式

| 类型 | 表示 | 示例 |
|------|------|------|
| 稀疏关键点 | (x, y, z, θ) 抓取点 | GraspNet |
| 密集热力图 | pixel-wise affordance heatmap | Where2Act |
| 参数化曲面 | 可抓取的几何区域 | CustomNet |
| 语义分割 | 逐像素"可抓/可推/可开"标签 | Demo2Vec |

### 主流方法

| 方法 | 输入 | 输出 | 特点 |
|------|------|------|------|
| 3D 几何分析 | 点云 | 抓取候选 | 经典、可解释 |
| 2D → 3D 投影 | RGB-D | 3D affordance | 结合语义 |
| 端到端学习 | RGB/点云 | 动作参数 | 数据驱动 |
| 大模型/VLM | 图像+文本 | 文本+位置 | 零样本泛化 |
| NeRF/3DGS | 多视角 | 3D affordance | 新视角合成 |

## 与 VLA 和 Grasp 的关系

```
Affordance ─→ 告诉 VLA "哪里可以做动作"
VLA ────────→ 根据 affordance 生成动作序列
Grasp ─────→ affordance 中"抓"这个动作子集

三者关系:
  Affordance = 广义的动作可能性（抓/推/按/拉/放/开...）
  Grasp = Affordance 的一种（专注于抓取）
  VLA = 利用 affordance 信息做决策和动作生成
```

## 关键论文

| 论文 | 年份 | 贡献 | 笔记 |
|------|------|------|------|
| Where2Act | 2021 | 像素级 affordance 预测 | [[]] |
| VAT-MART | 2022 | 视觉 affordance 与工具使用 | [[]] |
| IFR-Explore | 2023 | 交互式 affordance 探索 | [[]] |
| ManiGaussian | 2024 | 3DGS + affordance | [[]] |
| Robo-ABC | 2025 | 大规模 affordance 数据 | [[]] |

## 论文精读

👉 [[论文精读/|Affordance 论文精读目录]]

## 实验项目

👉 [[实验项目/|Affordance 实验记录]]

## 成员贡献

👉 [[成员笔记/|Affordance 成员笔记]]

---

> 💡 想在这个方向分享？在 `03-Affordance/论文精读/` 下新建笔记，使用 [[../../templates/论文笔记模板|论文笔记模板]]。
