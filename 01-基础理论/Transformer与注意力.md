# Transformer 与注意力机制

> 当代 AI 的核心引擎——VLA 的每一部分都离不开它

## 核心公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

## 架构组成

### Multi-Head Self-Attention
- 多组 Q/K/V 并行计算 → 每个头关注不同特征
- 拼接后线性投影

### Position Encoding
- Transformer 本身不具备位置感知 → 需要注入位置信息
- 可学习的 Position Embedding 或正余弦编码

### Feed-Forward Network (FFN)
- 两层 MLP + GELU
- 提供非线性和记忆容量

### Layer Normalization
- Pre-norm vs Post-norm
- 现代架构多用 Pre-norm（训练更稳定）

### Residual Connection
- 每个子层都有残差连接
- 让梯度直通底层，训练更深网络

## VLA 中的应用

```
RT-1: Transformer Encoder 处理压缩 token → 输出动作
RT-2: LLM 的 Transformer 自回归生成动作 token
OpenVLA: Transformer Decoder (LLaMA) + 视觉 token 前置
ACT: Transformer Encoder-Decoder (条件 VAE 架构)
```

## 关键创新点
- **TokenLearner** (RT-1): 从 486 个 visual token 压缩到 8 个
- **Cross-Attention**: 多模态融合的核心——视觉特征 attend 语言特征

## 自检问题
- [ ] 我能手写 Self-Attention 的完整代码
- [ ] 我理解 Q/K/V 各自的物理含义
- [ ] 我知道为什么需要除以 √d_k
- [ ] 我能解释 Multi-Head 的意义

## 关联笔记
- [[深度学习基础]] — Transformer 的前置知识
- [[大语言模型与微调]] — Transformer 的 scale-up 版本
- [[计算机视觉与ViT]] — Transformer 如何看图像
- [[多模态融合架构]] — Cross-Attention 实现模态融合
- [[VLA模型总览]] — Transformer 在 VLA 中的实际使用
