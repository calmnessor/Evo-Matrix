# 计算机视觉与 ViT

> VLA 中 V = Vision——机器人要"看见"世界才能行动

## 范式转变: CNN → ViT

| 特性 | CNN | ViT |
|------|-----|-----|
| 感受野 | 局部 → 逐渐全局 | 天然全局 |
| 归纳偏置 | 强（局部、平移不变） | 弱（数据驱动） |
| 数据需求 | 中等 | 大规模 |
| VLA 采用 | 逐渐被替代 | 主流选择 |

## ViT 核心流程

```
图像 [3, 224, 224]
  → Patch Embedding: 切成 14×14 个 16×16 patch
  → 每个 patch 线性投影到 d_model (如 768)
  → 加 CLS Token + Position Embedding
  → Transformer Blocks × N
  → 输出特征
```

## CLIP / SigLIP — VLA 的视觉标配

### CLIP: 对比学习对齐视觉-语言
- 图像编码器 + 文本编码器 → 共享嵌入空间
- N×N 相似度矩阵 → 对角线最大化
- 让模型理解语义（而不仅是标签）

### SigLIP: VLA 首选
- 用 Sigmoid 替代 Softmax → 不依赖 batch size
- OpenVLA 使用 `google/siglip-so400m-patch14-384`
- 架构: ViT-L, 27 层, 1152 hidden, patch=14

## 在 VLA 中的角色

```
摄像头图像 → SigLIP 视觉编码器 → patch 特征
                                    ↓
文本指令 → LLM Tokenizer → token embedding
                                    ↓
                    拼接/Cross-Attention → Transformer
                                    ↓
                              动作输出
```

## 自检问题
- [ ] 我能画出 ViT 的完整数据流
- [ ] 我理解 Patch Embedding 的原理
- [ ] 我知道 CLIP 的对比学习训练方式
- [ ] 我知道 SigLIP 相对于 CLIP 的改进
- [ ] 我分析了 OpenVLA 的视觉编码器配置

## 关联笔记
- [[Transformer与注意力]] — ViT 的引擎
- [[深度学习基础]] — CNN 基础
- [[VLA模型总览]] — 视觉编码器在 VLA 中如何使用
- [[多模态融合架构]] — 视觉特征如何与语言融合
