# 分层 VLA（大小脑）

> System-1 / System-2 架构 — 受 Kahneman《思考，快与慢》启发，将 VLA 分为"快系统"（直觉式动作执行）和"慢系统"（推理式任务规划），兼顾实时控制与复杂推理。

## 核心思想

```
传统 VLA:                       分层 VLA:
  感知 → [一个大模型] → 动作       感知 → System-2 (慢, ~5Hz): "我该做什么?"
                                            ↓ 子任务/目标
                                       System-1 (快, ~50-100Hz): "怎么做到?"
                                            ↓
                                          动作
```

## 为什么需要分层？

| | 纯 VLA (如 π₀, OpenVLA) | 分层 VLA |
|------|------|------|
| 推理深度 | 浅（动作级） | 深（任务级+动作级） |
| 长周期任务 | 困难 | System-2 分解子任务 |
| 实时性 | 受限于 VLM 推理速度 | System-1 独立高速运行 |
| 纠错能力 | 弱 | System-2 可监控+修正 |
| 语言遵循 | 直接映射语言→动作 | System-2 先理解→再交给 System-1 |

## 论文列表

| # | 论文 | 日期 | 核心贡献 | 笔记 |
|---|------|------|----------|------|
| 1 | Hi Robot | 2025.02 | 首个 PI 分层架构：VLM System-2 + VLA System-1 | [[Hi Robot]] |
| 2 | Fast-in-Slow (FiS-VLA) | 2025.09 | System-1 嵌入 VLM 内部，117Hz 控制 | [[Fast-in-Slow]] |
| 3 | Hume | 2025.05 | 价值引导 System-2 思维 + 级联去噪 System-1 | [[Hume]] |

## 技术路线对比

| | Hi Robot | Fast-in-Slow | Hume |
|------|------|------|------|
| **System-2 频率** | ~3 Hz | ~30 Hz | ~4 Hz |
| **System-1 频率** | ~50 Hz | **117 Hz** | 90 Hz |
| **System-2 实现** | 独立 VLM | VLM 末层复用 | VLM + Q-value head |
| **System-1 实现** | 独立 Action Expert | VLM 浅层复用 | 级联去噪模块 |
| **参数共享** | 部分 | 大部分 | 部分 |
| **特色** | 开山之作 | 嵌入 VLM 内部 | 价值引导+人形 |
| **硬件** | Franka, ALOHA | Franka | WidowX, Franka, **AgiBot G-1** |

## 推荐阅读顺序

1. 先读 **Hi Robot** — 理解分层 VLA 的基本概念和 PI 的方案
2. 再读 **Fast-in-Slow** — 理解如何更优雅地实现双系统（VLM 内部复用）
3. 然后 **Hume** — 理解价值函数如何引导 System-2 推理

## 相关方向

- [[../../VLA方向综述|VLA 方向综述]] — VLA 全景
- [[../pi系列/README|π 系列论文]] — PI 的单系统 VLA 基础
