# Hume: Introducing System-2 Thinking in Visual-Language-Action Model

> 2025 — 用学习的价值函数（Q-value）引导 System-2 在多个候选方案中择优，配合级联去噪 System-1 执行——首个在人形机器人上验证的分层 VLA。

## 论文信息

- **标题**: Hume: Introducing System-2 Thinking in Visual-Language-Action Model
- **日期**: 2025 年 5 月
- **arXiv**: [2505.21432](https://arxiv.org/abs/2505.21432)
- **机构**: SJTU, Shanghai AI Lab, Zhejiang Univ, AgiBot
- **作者**: Haoming Song, Delin Qu, Yuanqi Yao 等

## 一句话总结

Hume 在 VLA 中引入 **System-2 思维**——不仅直接输出动作，而是先"多想一想"：生成多个动作候选 → 用学习的价值函数打分 → 选出最优 → System-1 级联去噪执行。在 WidowX、Franka 和 AgiBot G-1 人形机器人上均验证有效。

## 核心动机

### System-2 到底该做什么？

```
Hi Robot:  System-2 → 子任务描述 (文本) → System-1 执行
            ↑ 文本粒度粗，丢失空间信息

Fast-in-Slow: System-2 → 语义 token → System-1 直接读
              ↑ 不需要显式规划，隐式在 token 中

Hume:  System-2 → 采样多个动作提案 → 价值评估 → 选最优
       ↑ 显式的"多想一想"，可解释、可验证
```

### 为什么需要价值函数引导？

VLA 直接输出动作 = **直觉反应**（Kahneman System-1）。有时直觉是错的：
- 抓取时该从左边还是右边？两个都合理但成功率不同
- 窄缝中该先旋转还是先平移？顺序很关键
- 面对新物体时，最像哪个见过的物体？

Hume 的方案：**System-2 生成 K 个候选动作 → Q-value 打分 → 选最优**。

## 架构

### 双系统设计

```
输入: 图像 + 语言指令 + 状态
  ↓
┌─────────────────────────────────────────┐
│ System-2 (慢推理, ~4 Hz)                  │
│                                           │
│  VLM Encoder → 理解场景和指令              │
│       ↓                                   │
│  Action Proposal Head                     │
│  → 采样 K 个候选动作 {a₁, a₂, ..., a_K}    │
│       ↓                                   │
│  Q-Value Head (离线 RL 训练)               │
│  → Q(s, a₁), Q(s, a₂), ..., Q(s, a_K)    │
│       ↓                                   │
│  选择 a* = argmax Q(s, a)                  │
│       ↓                                   │
│  输出: 最优动作提案 a*                      │
└─────────────────────────────────────────┘
  ↓ a* (最优提案)
┌─────────────────────────────────────────┐
│ System-1 (快执行, 90 Hz)                  │
│                                           │
│  级联去噪模块 (Cascaded Denoising)         │
│  a* 作为初始条件 + 当前高频观测             │
│  → 多步去噪 → 精细化的高频动作              │
│  → 直接发送到机器人控制器                  │
└─────────────────────────────────────────┘
  ↓ 动作
```

### Q-Value 的训练

```python
# 离线 RL 训练 Q-value head
# 使用历史成功/失败数据

for (o, a, success) in offline_data:
    # 目标 Q 值: 成功=1, 失败=0
    target = 1.0 if success else 0.0

    # TD 风格也可以:
    # target = r + γ * V(o_next)

    q_pred = Q_head(VLM_encode(o), a)
    loss = MSE(q_pred, target)
```

关键是**不需要在线 RL**——仅从历史遥操作数据中的成功/失败信号学一个"什么是好动作"的判断能力。

### 级联去噪 (Cascaded Denoising)

```
System-2 输出: 粗略的动作方向 (低频, ~4Hz)
  ↓
System-1 级联去噪:
  Step 1: 去噪 (粗糙 → 中等)  ~2ms
  Step 2: 去噪 (中等 → 精细)  ~2ms
  Step 3: 去噪 (精细 → 精确)  ~2ms
  ↓
高频精细动作 (~90Hz)
```

比纯 Flow Matching 从零采样更高效——从 System-2 的近似出发，只需少量去噪步骤。

## 关键结果

| 平台 | 任务 | 基线 (纯 System-1 VLA) | Hume |
|------|------|------|------|
| LIBERO (仿真) | 多任务 | ~90% | **98.6%** |
| WidowX (真机) | 桌面操作 | ~80% | **91%** |
| Franka | 灵巧操作 | — | 显著提升 |
| **AgiBot G-1** (人形) | 叠衣 | — | **88%** |

### 为什么价值引导有效？

Hume 的消融实验发现：
- 随机选候选动作：性能接近基线
- Q-value 引导选择：大幅提升
- **价值函数学会了"什么是好的动作"**——即使它没见过当前确切场景

## 贡献

1. **价值引导 System-2**：首次在 VLA 中引入离线 RL 训练的 Q-value 来择优
2. **级联去噪**：从 System-2 提案出发去噪，效率高于从零生成
3. **90Hz System-1**：级联去噪保证高频实时性
4. **人形验证**：AgiBot G-1 上的 88% 叠衣成功率，分层 VLA 首次在人形上展示

## 局限性

- Q-value 训练依赖成功/失败标签（需要可靠的任务判别器）
- K 个候选的采样增加 System-2 推理开销（K×）
- 在极度新颖场景下 Q-value 可能不准确

## 自检问题

- [ ] 我理解 Hume 的 Q-value head 如何训练（离线 RL vs 在线 RL）
- [ ] 我能解释级联去噪相比从零 Flow Matching 的优势
- [ ] 我知道为什么 K 个候选中 Q-value 择优优于直接输出
- [ ] 我能对比 Hume、Hi Robot、FiS 三种分层方案

## 关联笔记

- [[Hi Robot]] — 分层 VLA 的开山之作，System-2 输出文本子任务
- [[Fast-in-Slow]] — VLM 内部分层，参数复用
- [[../pi系列/π-0.6-RECAP|π*₀.₆ RECAP]] — 同样使用价值函数，但用于 RL 自进化
