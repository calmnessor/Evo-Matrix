# Transformer 与注意力机制

> 当代 AI 的核心引擎——VLA 的每一部分都离不开它。理解 Transformer，才算真正入门现代深度学习。

## 为什么 Transformer 改变了 AI？

在 Transformer (2017) 之前，处理序列的标准方案是 RNN/LSTM。RNN 的问题是：第 100 个 token 要看第 1 个 token 的信息，要穿过 99 层循环——梯度消失严重，而且无法并行（必须等 t-1 算完才能算 t）。

Transformer 用 **Self-Attention** 一招解决了这两个问题：
- **长程依赖**：每个 token 直接 attend 到所有其他 token（$O(1)$ 路径长度）
- **并行化**：所有 token 的 attention 可以同时计算

这使得 Transformer 成为第一个能高效扩展到数千亿参数的架构。

---

## 1. 核心公式与直觉

### 1.1 Self-Attention：三个矩阵的故事

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Q、K、V 的物理含义**（用人话理解）：

```
Q (Query，查询):  "我想找什么样的信息？"——来自当前 token
K (Key，键):     "我包含什么信息？"——来自所有 token
V (Value，值):   "我的实际内容是什么？"——所有 token 的 value

流程:
  Q @ K^T → 相似度矩阵 (哪两个 token 相关?)
  /√dk → 防止点积过大导致 softmax 梯度消失
  softmax → 转为注意力权重 (每行和为 1)
  @ V → 加权求和 = "我应该关注哪些 token 的内容"
```

**一个直观例子**：句子 "The cat sat on the mat because it was tired"

当 $Q$ = "it" 的 Query，$K$ = 所有词的 Key：
- "it" 和 "cat" 的相似度 = 0.8（高）
- "it" 和 "mat" 的相似度 = 0.3（中）
- "it" 和 "sat" 的相似度 = 0.1（低）
→ 权重最高的 V 是 "cat"，所以模型知道 "it" 指代的是 "cat"

### 1.2 为什么除以 $\sqrt{d_k}$？

点积 $QK^T$ 的方差 $\propto d_k$。当 $d_k$ 很大（如 128），点积值可能很大 → softmax 输出接近 one-hot → 梯度趋近 0。

除以 $\sqrt{d_k}$ 将方差缩回 $1$，让 softmax 保持在"软"的区间。

### 1.3 Multi-Head Attention

**为什么需要多个头？**

单头 attention 平均了不同语义关系。多头让不同头关注不同类型的关系：

```
Head 1: 关注语法关系 (主谓宾)
Head 2: 关注指代关系 (代词 → 先行词)
Head 3: 关注语义相似 (同义词)
Head 4: 关注位置相邻 (局部上下文)
...
```

**实现**：把 $d_{model}$ 切分为 $h$ 个 $d_k = d_{model}/h$ 的子空间，每个头在自己的子空间做 attention，最后 concat + 线性投影。

```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) · W^O

其中 head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

**典型配置**：LLaMA-7B: $d_{model}=4096$, $h=32$, $d_k=128$

---

## 2. 完整架构

### 2.1 Transformer Block

```
输入: x ∈ R^{n×d_model}

Block(x):
  # 自注意力子层
  attn_out = MultiHeadSelfAttention(LayerNorm(x))  ← Pre-norm
  x = x + attn_out                                  ← Residual
  
  # 前馈子层  
  ff_out = FFN(LayerNorm(x))
  x = x + ff_out                                    ← Residual
  
  return x
```

### 2.2 三个关键设计

| 组件 | 作用 | 不用会怎样 |
|------|------|----------|
| **Residual Connection** | 梯度直通底层，训练更深网络 | 深层梯度消失，无法训练 >10 层 |
| **Layer Normalization** | 稳定每层的输入分布 | 训练不稳定，loss 震荡 |
| **Position Encoding** | 注入位置信息（Transformer 本身看不到顺序） | "A 打 B"和"B 打 A"变成同一句话 |

### 2.3 三种 Transformer 变体

| 类型 | 代表 | Attention 模式 | 用途 |
|------|------|---------------|------|
| **Encoder-only** | BERT | 双向 (每个 token 看到所有 token) | 理解任务 (分类、提取) |
| **Decoder-only** | GPT/LLaMA | Causal (只能看上文) | 自回归生成 |
| **Encoder-Decoder** | T5/BART | Encoder 双向 + Decoder Causal + Cross-Attention | 翻译、摘要 |

---

## 3. 从零实现 Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        # Q/K/V 投影矩阵 (合并在一个矩阵中提高效率)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, N, D = x.shape  # batch, seq_len, d_model
        
        # 1. 投影 Q/K/V: [B, N, 3*D] → 3 × [B, n_heads, N, d_k]
        qkv = self.W_qkv(x).reshape(B, N, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, N, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 2. 计算注意力: [B, n_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Causal Mask (Decoder 模式)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # 4. 加权求和: [B, n_heads, N, d_k]
        out = attn @ v
        
        # 5. 合并多头: [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.W_o(out)


# 使用示例
attn = MultiHeadSelfAttention(d_model=512, n_heads=8)
x = torch.randn(2, 64, 512)  # [batch=2, seq_len=64, d_model=512]
output = attn(x)
print(f"输入: {x.shape} → 输出: {output.shape}")  # [2, 64, 512]
```

---

## 4. 视觉 Transformer 变体概览

| 模型 | 年份 | 核心创新 | VLA 使用情况 |
|------|------|---------|------------|
| **ViT** | 2020 | Patch Embedding + Transformer | 基础架构 |
| **CLIP** | 2021 | 对比学习对齐图文 | 泛用的视觉语义特征 |
| **SigLIP** | 2023 | Sigmoid 替代 Softmax | OpenVLA 标配 |
| **DINOv2** | 2023 | 自监督学习，强语义 | 新兴 VLA 选择 |
| **VideoMAE** | 2022 | 视频自监督 | 时序动作识别 |

---

## 5. 新手学习路线

### 第一阶段：理解 attention 本质 (1-2 天)
1. 读懂上面 Self-Attention 的代码实现，一行一行手写在纸上
2. 阅读 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 最经典的可视化教程
3. 用一张小矩阵 (3×3) 手动计算一遍 attention：Q @ K^T → /√dk → softmax → @ V

### 第二阶段：上手代码 (2-3 天)
1. 从 `torch.nn.MultiheadAttention` 开始，理解 API
2. 用 `transformers` 库加载一个 GPT-2 小模型，打印每层的 attention 权重
3. 修改上面的代码：加 Causal Mask、加 Cross-Attention

### 第三阶段：VLA 关联 (2-3 天)
1. 读 RT-1 论文的 TokenLearner 部分
2. 读 OpenVLA 代码中的模型定义（`prismatic.py`）
3. 理解：visual tokens 是如何拼接进 LLaMA 的 embedding 序列的

---

## 6. 推荐论文与资源

### 必读论文

| # | 论文 | 关键词 | 理由 |
|---|------|--------|------|
| 1 | **Attention Is All You Need** (1706.03762) | Transformer 原论文 | 一切的原点 |
| 2 | **ViT** (2010.11929) | Patch Embedding | Transformer 如何看图像 |
| 3 | **CLIP** (2103.00020) | 对比学习 + 视觉-语言 | VLA 视觉编码器的基础 |
| 4 | **RT-1** (2212.06817) | TokenLearner | Transformer 在机器人上的首次大规模应用 |
| 5 | **LLaMA** (2302.13971) | 开源 LLM | OpenVLA 的 backbone 架构 |

### 教程资源

- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**：必看，一图胜千言
- **[The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)**：逐行注释的完整实现
- **[3Blue1Brown: Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc)**：最直观的视频讲解
- **[Andrej Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)**：从零开始写一个 GPT

---

## 7. 自检问题

### 基础关
- [ ] 我能手写 Self-Attention 的前向传播（伪代码即可）
- [ ] 我能解释 Q/K/V 各自的物理含义
- [ ] 我知道为什么需要除以 $\sqrt{d_k}$
- [ ] 我能解释 Multi-Head 的意义
- [ ] 我理解 Residual Connection 为什么让深层网络可以训练

### 进阶关
- [ ] 我知道 Causal Mask 如何实现 (上三角 mask)
- [ ] 我能解释 Pre-norm 和 Post-norm 的区别和选择
- [ ] 我能区分 Self-Attention 和 Cross-Attention 的使用场景
- [ ] 我理解 TokenLearner 如何压缩 token 数量
- [ ] 我能解释 Position Encoding 三种方案 (Sinusoidal/Learnable/RoPE) 的差异

### 实战关
- [ ] 我能用 PyTorch 实现一个完整的 Transformer Block
- [ ] 我能加载预训练的 ViT 并提取图像特征
- [ ] 我能看懂 OpenVLA 源码中的模型定义部分
- [ ] 我知道在 VLA 中 visual token 如何与 text token 拼接

---

## 关联笔记

- [[位置编码与归一化]] — Position Encoding 详解 + Pre/Post-norm
- [[Cross-Attention与VLA应用]] — 多模态融合的核心 + VLA 架构
- [[../深度学习基础/深度学习基础|深度学习基础]] — 反向传播、梯度下降等 Transformer 训练基础
- [[../大语言模型与微调/大语言模型与微调|大语言模型与微调]] — Transformer 的 scale-up 版本 + LoRA 微调
- [[../计算机视觉与ViT/计算机视觉与ViT|计算机视觉与ViT]] — Transformer 如何应用于视觉
