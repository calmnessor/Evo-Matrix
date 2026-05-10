# ACT (Action Chunking Transformer)

> 动作分块 + 条件 VAE → 2023 年机器人学习的标志性方法

## 一句话

ACT = CVAE + Action Chunking + Transformer

把动作生成变为"编码-采样-解码"的过程。

## 核心三件套

### 1. CVAE (Conditional VAE)

```
传统 BC:  obs → policy → deterministic action
ACT:      obs → encoder → latent z ~ N(μ, σ) → decoder → action
```

- **Encoder** (训练时): 看到"标准答案"动作 → 生成 latent z
- **Decoder** (训练+推理): 从 obs + z 重建动作
- **KL Loss**: 约束 z 的分布接近标准正态

### 2. Action Chunking

```
预测: ████████████████  16 步动作一口气生成
执行: ████████          取前 8 步执行
更新:         ████████████████  再用新观测预测 16 步（含 8 步重叠）

→ 执行频率可以大于推理频率
→ 动作平滑（重叠部分取平均）
```

### 3. Transformer Encoder-Decoder

```
Encoder: 输入当前观测 → 编码场景
StyleDrop: z 通过 StyleDrop 注入 Decoder
Decoder: Cross-Attention to Encoder + z → 输出动作序列
```

## 架构

```
训练阶段:
  图像序列 → CNN → Encoder → obs_feat
                                         ↘
  真实动作序列 → Action Encoder → (μ, σ) → z
                                               ↗
  z + obs_feat → Decoder → 重建动作
  Loss = MSE(action, reconstructed) + β * KL( N(μ,σ) || N(0,I) )

推理阶段:
  图像序列 → CNN → Encoder → obs_feat
                                    ↘
  标准高斯采样 → z → obs_feat + z → Decoder → 动作序列
```

## 为什么有效？

1. **Action Chunking → 减少累积误差**: 一次预测多步，不像 BC 步步误差累积
2. **VAE → 平滑动作**: latent z 的随机性提供多样性
3. **Temporal ensemble → 更平滑**: 重叠取平均，减少抖动

## ACT vs Diffusion Policy

| 维度 | ACT | Diffusion Policy |
|------|-----|-----------------|
| 生成方式 | VAE Decoder 一步 | 扩散去噪 N 步 |
| 推理速度 | 快（1 step） | 慢（N steps, 可用 DDIM） |
| 动作多峰 | 有限（VAE 的 z 采样） | 强（扩散天然多峰） |
| 动作平滑 | 好（Chunking + overlapping） | 很好 |
| 训练复杂度 | 中（VAE + KL） | 中（去噪预测） |

## 自检问题
- [ ] 我理解 CVAE 和普通 AE 的区别
- [ ] 我能解释 KL Loss 在 VAE 中的作用
- [ ] 我知道 Action Chunking 如何减少累积误差
- [ ] 我能对比 ACT 和 Diffusion Policy 的优劣

## 关联笔记
- [[扩散策略]] — 同期的另一种动作生成方案
- [[行为克隆]] — 最简单的替代方案
- [[模仿学习]] — ACT 的理论基础
- [[VLA模型总览]]
- [[机器人运动学]] — 关节空间控制 vs 笛卡尔空间
