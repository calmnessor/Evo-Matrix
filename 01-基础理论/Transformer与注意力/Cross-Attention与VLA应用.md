# Cross-Attention 与 VLA 应用

> 多模态融合的核心——视觉和语言两个世界如何"对话"。

---

## 1. Cross-Attention：多模态融合的核心

在 VLA 中，视觉特征和语言特征需要"交流"。**Cross-Attention** 就是桥梁。

```python
# Self-Attention: Q, K, V 来自同一来源
self_attn = Attention(Q=x, K=x, V=x)

# Cross-Attention: Q 来自一个模态，K/V 来自另一个模态
cross_attn = Attention(Q=visual_features,  # "用视觉提问"
                       K=text_features,    # "去语言中找答案"  
                       V=text_features)
```

**物理直觉**：Cross-Attention 让一个模态（Q）去"查询"另一个模态（K, V）。比如视觉特征（Q）问"我应该关注指令中的哪些词？"，语言特征（K/V）回答。

---

## 2. Transformer 在 VLA 中的实际应用

```
RT-1 (Google, 2022):
  图像序列 → CNN → TokenLearner (压缩 486→8 token)
  8 visual tokens + 动作历史 → Transformer Encoder → 离散化动作
  
  关键创新: TokenLearner 大幅压缩 token 数量

RT-2 (Google, 2023):
  图像 → ViT → visual tokens
  文本 → LLM Tokenizer → text tokens
  concat → PaLI-X/PaLM-E Decoder → 自回归生成动作 token
  
  关键创新: 动作离散化为 token，与文本 token 共存

OpenVLA (Stanford, 2024):
  图像 → SigLIP → visual tokens [B, 256, 1152]
  投影层压缩: [B, 256, 1152] → [B, 256, 4096]
  文本 → LLaMA Embedding → [B, L, 4096]
  concat → LLaMA Decoder → 离散动作 bin
  
  关键创新: 开源 + LoRA 轻量适配

ACT (模仿学习经典):
  图像 + 关节角 → Encoder(压缩为 latent z)
  z + 当前关节角 → Decoder(Cross-Attn) → 动作序列 (Chunk)
  
  关键创新: 用 Transformer 做条件 VAE，一次输出多个未来动作
```

---

## 3. 三种多模态融合模式

### 3.1 Token Concatenation (拼接)

```
[视觉 token] [文本 token] [动作 token] → Transformer Decoder
```

- 代表：RT-2, OpenVLA, LLaVA
- 优点：简单，不需要额外结构
- 缺点：token 数多 → 计算量大（$O(N^2)$ attention）

### 3.2 Cross-Attention Gate

```
文本 token → Self-Attention → FFN → Cross-Attention(Q=文本, K/V=视觉) → FFN → ...
```

- 代表：Flamingo/OpenFlamingo
- 优点：可以和现有 LLM 更好结合
- 缺点：架构更复杂

### 3.3 TokenLearner / Perceiver (压缩)

```
大量视觉 token → TokenLearner → 少量 token → concat → LLM
```

- 代表：RT-1
- 优点：大幅减少 token 数 → 加速
- 缺点：可能丢失细粒度信息

---

## 4. Token 拼接的细节（OpenVLA 为例）

```
Step 1: 图像 [3, 384, 384] → SigLIP ViT
        → visual tokens [B, 729, 1152]  # 729 = (384/14)²

Step 2: 投影到 LLM 空间
        → MLP: 1152 → 4096
        → visual tokens [B, 729, 4096]

Step 3: 文本 "pick up the red cup" → LLaMA Tokenizer
        → text tokens [B, L, 4096]  # L ≈ 8-12 tokens

Step 4: 拼接
        → input = [visual | text] = [B, 729+L, 4096]

Step 5: LLaMA Decoder 自回归生成
        → 预测动作 bin [B, 1, 256]  # 256 个离散化动作 bin
```

**关键问题**：因果注意力下，visual token 能看 visual token，text token 能看 visual + text。这意味着视觉上下文影响语言理解，但语言不影响视觉特征——这合理吗？

---

## 5. 关联笔记

- [[Transformer与注意力]] — Self-Attention 基础
- [[位置编码与归一化]] — RoPE 在 VLA 中的作用
- [[../视觉基础模型/ViT/ViT概述|视觉基础模型]] — 视觉编码器详解
- [[../大语言模型与微调/大语言模型与微调|大语言模型与微调]] — LLM Backbone 详解
- [[../../02-VLA/VLA模型总览|VLA模型总览]] — VLA 各模型架构全景对比
