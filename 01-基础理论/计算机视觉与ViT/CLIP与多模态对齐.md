# CLIP 与多模态对齐

> VLA 的视觉编码器选型——CLIP 和 SigLIP 如何让模型"看"懂"语言"描述的物体。

---

## 1. CLIP：对比学习对齐视觉和语言

### 训练方式

```
Batch 内有 N 个 (图像, 文本) 对：

图像: I₁  I₂  ... I_N    文本: T₁  T₂  ... T_N
      ↓  ViT                ↓  Text Transformer
视觉特征: v₁  v₂  ... v_N    文本特征: t₁  t₂  ... t_N

计算 N×N 相似度矩阵:
  S[i,j] = v_i · t_j / τ  (τ = 可学习温度)

训练目标: 最大化对角元素 (正确配对)，最小化非对角元素
         这就是 "对比学习" — 让配对的靠近，不配对的远离
```

**CLIP 给 VLA 带来了什么？**
- 零样本理解："没见过训练集中的杯子"也能通过文本描述识别
- 开放词汇：不是"分类为 class_37"，而是"这是红色马克杯"
- 语义视觉特征：不只是"看起来像杯子"，而是"这是装热饮的容器"

---

## 2. SigLIP：VLA 的标配视觉编码器

**相比 CLIP 的改进**：把对比损失中的 softmax (依赖 batch size) 换成独立的 sigmoid：

- CLIP: `loss = -log(exp(S[i,i]) / Σ_j exp(S[i,j]))` ← 需要大 batch 提供足够负样本
- SigLIP: `loss = -log(σ(S[i,i])) - Σ_{j≠i} log(1-σ(S[i,j]))` ← 每个 pair 独立判断

**结果**：SigLIP 不依赖 batch size，小 batch 也能稳定训练。

**OpenVLA 配置**：`google/siglip-so400m-patch14-384`
- 训练图像大小：384×384
- Patch size：14×14 → 729 个 visual token
- Hidden dim：1152
- 通过两层 MLP 投影到 LLaMA 的 4096 维空间

```python
# OpenVLA 中的视觉投影器
projector = nn.Sequential(
    nn.Linear(1152, 4096),  # SigLIP hidden → LLaMA hidden
    nn.GELU(),
    nn.Linear(4096, 4096),
)
```

---

## 3. 视觉编码器选型对比

| 编码器 | 预训练方式 | 参数量 | 输出 token 数 | 适合场景 |
|--------|----------|--------|-------------|---------|
| **SigLIP SO400M** | 图文对比 (sigmoid) | 400M | 729 | OpenVLA 标配 |
| **CLIP ViT-L** | 图文对比 (softmax) | 300M | 256 | 通用 VLA，零样本强 |
| **DINOv2 ViT-L** | 自监督 | 300M | 256 | 强调几何/空间理解 |
| **EfficientNet-B3** | 监督分类 | 12M | — (特征向量) | 轻量 / RT-1 风格 |
| **InternViT-300M** | 图文 + 自监督 | 300M | 可变 | 中文场景友好 |

### 选型决策树

```
需要零样本语言理解？
├── 是 → 需要轻量部署？
│        ├── 是 → EfficientNet + CLIP text
│        └── 否 → SigLIP (推荐) 或 CLIP ViT-L
└── 否 → 只需要视觉几何理解？
         └── 是 → DINOv2
```

---

## 关联笔记

- [[计算机视觉与ViT]] — ViT 架构 + VLA 视觉流程
- [[DINOv2与自监督视觉]] — 自监督替代方案
- [[../Transformer与注意力/Cross-Attention与VLA应用|Cross-Attention与VLA应用]] — 视觉 token 如何注入 LLM
