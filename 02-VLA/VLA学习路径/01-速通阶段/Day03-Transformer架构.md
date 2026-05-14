# ⚡ Day 3: Transformer 架构原理（8 小时）

> **口号**: "Transformer 是所有现代 AI 的共同语言，今天必须啃下来！"  
> **目标**: 从零理解 Self-Attention，手写 Mini Transformer，精读原论文  
> **为什么重要**: VLA = Vision Transformer + Language Model + Action Head，Transformer 是三者共同的基础

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | Attention 的直觉理解 | 1h | [ ] |
| 2 | Self-Attention 逐行拆解 | 1.5h | [ ] |
| 3 | Multi-Head Attention | 1h | [ ] |
| 4 | Transformer Block 完整结构 | 1h | [ ] |
| 5 | Position Encoding | 0.5h | [ ] |
| 6 | 动手：手写 Mini Transformer | 1.5h | [ ] |
| 7 | 论文精读 + 费曼挑战 | 1.5h | [ ] |

---

## 模块 1: Attention 的直觉理解（1 小时）

### 1.1 类比先行 — 10min

```
你在看一张杂乱的桌子找钥匙:

查询 Query  = "我想要找钥匙"
键   Key    = 桌上每个物品的"特征标签"（颜色、形状、位置）
值   Value  = 每个物品本身

Attention 做的事:
  Q 与每个 K 计算相似度 → 得到关注权重
  用权重对 V 加权求和 → 得到"你应该关注的东西"
```

### 1.2 从 RNN 到 Attention — 20min

```
RNN 的问题（Transformer 论文的背景）:
  "The dog, which ate the apple, was hungry."
  处理 "was" 时需要知道主语是 "dog"（隔了 4 个词）
  RNN: 必须一步步传，信息会被稀释
  Attention: 每个词可以直接"看到"所有其他词

这就是 Self-Attention 的核心优势: 全局视野、并行计算、长距离依赖
```

### 1.3 VLA 中的 Attention 视角 — 20min

```python
# 图像 patch 作为 token
# "请拿起红色积木" 作为 token
# → 所有 token 互相关注

# Attention 让模型能做到:
# "红色" (语言) ↔ 图像中红色积木的位置 (视觉)
# 当前机器人姿态 ↔ 下一步应该怎么动
```

**自检**: 用一句话解释 Attention 和传统 CNN 的全连接层有什么本质不同？

---

## 模块 2: Self-Attention 逐行拆解（1.5 小时）

> ⚠️ 以下是今日最核心内容，每行代码都要理解！

### 2.1 从头推导 — 30min

```
输入: X ∈ R^(n × d)   （n 个 token，每个 d 维）

步骤 1: 生成 Q, K, V
  Q = X @ W_Q    (n × d_k)
  K = X @ W_K    (n × d_k)
  V = X @ W_V    (n × d_v)

步骤 2: 计算注意力分数
  Scores = Q @ K^T / √d_k    (n × n)
  ↑ 除以 √d_k 是为了防止点积过大导致 softmax 梯度消失

步骤 3: Softmax 归一化
  Attention_Weights = softmax(Scores, dim=-1)    (n × n)
  ↑ 每行是一个概率分布: 第 i 个 token 对各个 token 的关注程度

步骤 4: 加权求和
  Output = Attention_Weights @ V    (n × d_v)
  ↑ 每个输出 token 是所有输入 value 的加权和
```

### 2.2 手算一个例子 — 30min

```
设 n=3 (3个token), d=2 (2维特征)

X = [[1, 0],
     [0, 1],
     [1, 1]]

简化: W_Q = W_K = W_V = I (单位阵)
那么 Q = K = V = X

Scores = Q @ K^T:
  [[1,0],    [[1,0,1],     [[1, 0, 1],
   [0,1],  ×  [0,1,1]]  =   [0, 1, 1],
   [1,1]]                    [1, 1, 2]]

除以 √d_k = √2 ≈ 1.414:
  = [[0.71, 0,    0.71],
     [0,    0.71, 0.71],
     [0.71, 0.71, 1.41]]

Softmax (第一行):
  e^0.71 = 2.03, e^0 = 1, e^0.71 = 2.03
  和 = 5.06
  → [0.40, 0.20, 0.40]

Output = Attention_Weights @ V:
  第 1 个输出 = 0.40 * [1,0] + 0.20 * [0,1] + 0.40 * [1,1] = [0.80, 0.60]
```

**自检**: 自己选一组值，手算一遍完整的 Self-Attention！

### 2.3 PyTorch 实现 — 20min

```python
import torch
import torch.nn.functional as F

def self_attention(X, W_Q, W_K, W_V):
    """
    X: [batch, n_tokens, d_model]
    """
    Q = X @ W_Q  # [B, n, d_k]
    K = X @ W_K  # [B, n, d_k]
    V = X @ W_V  # [B, n, d_v]

    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # [B, n, n]
    attn_weights = F.softmax(scores, dim=-1)          # [B, n, n]
    output = attn_weights @ V                          # [B, n, d_v]

    return output, attn_weights

# 测试
B, N, D = 2, 5, 8
X = torch.randn(B, N, D)
W_Q = torch.randn(D, D)
W_K = torch.randn(D, D)
W_V = torch.randn(D, D)

out, weights = self_attention(X, W_Q, W_K, W_V)
print(f"Output shape: {out.shape}")        # [2, 5, 8]
print(f"Attention shape: {weights.shape}") # [2, 5, 5]
```

### 2.4 Attention 可视化理解 — 10min

```
对于 "the cat sat on the mat":

Attention Matrix (每个词关注哪些词):
         the  cat  sat  on  the  mat
the    [0.3, 0.1, 0.1, 0.1, 0.3, 0.1]
cat    [0.1, 0.5, 0.2, 0.1, 0.0, 0.1]
sat    [0.1, 0.2, 0.4, 0.2, 0.0, 0.1]
on     [0.1, 0.1, 0.2, 0.4, 0.1, 0.1]
the    [0.3, 0.0, 0.0, 0.1, 0.6, 0.0]
mat    [0.0, 0.1, 0.1, 0.1, 0.1, 0.6]

cat 最关注自己 (0.5) 和 sat (0.2) — 合理！
```

---

## 模块 3: Multi-Head Attention（1 小时）

### 3.1 为什么需要多头？ — 20min

```
单头 Attention: 只能学到一种"关注模式"
多头 Attention: 并行学多种关注模式

例如对于 VLA:
  Head 1: 关注空间位置关系（物体在哪里）
  Head 2: 关注语义匹配（"红色"对应的物体）
  Head 3: 关注时序关系（下一步应该做什么）
  Head 4: 关注安全约束（会不会碰到障碍物）
```

### 3.2 Multi-Head Attention 代码 — 30min

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # 输出投影

    def forward(self, x):
        B, N, D = x.shape

        # 1. 线性投影 + 分头
        # [B, N, D] → [B, n_heads, N, d_k]
        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 每个头独立计算 Attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # [B, n_heads, N, d_k]

        # 3. 合并多头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_O(out)

# 这个类在 VLA 代码中会被大量调用
```

**自检**: `transpose(1, 2)` 做了什么？为什么需要 `.contiguous()`？

---

## 模块 4: Transformer Block 完整结构（1 小时）

### 4.1 一个 Transformer Block = Attention + FFN — 30min

```
         ┌─────────────┐
    x ──→│ Multi-Head   │
    │    │ Attention    │
    │    └──────┬───────┘
    │           │
    └── + ──────┘  ← 残差连接 (Add)
    │
    │    ┌──────┴───────┐
    └──→ │  LayerNorm   │
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │  FeedForward  │  ← 2层 MLP: Linear → GELU → Linear
         └──────┬───────┘
                │
    ┌── + ──────┘  ← 残差连接
    │
    └──→ LayerNorm → 输出
```

### 4.2 完整实现 — 30min

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network: 两个线性层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),          # VLA 中用的激活函数
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-LN 风格（现代 VLA 的标准写法）
        # 1. Attention + 残差
        x = x + self.attention(self.norm1(x))
        # 2. FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x
```

### 4.3 完整 Transformer 串起来 — 15min

```python
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)  # 预测下一个 token

    def forward(self, token_ids):
        x = self.embedding(token_ids)       # [B, L, d_model]
        x = self.pos_encoding(x)            # 加入位置信息
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)              # [B, L, vocab_size]
```

---

## 模块 5: Position Encoding（0.5 小时）

### 5.1 为什么需要位置编码？ — 10min

```
Self-Attention 是置换不变的:
  "猫 坐 在 垫子 上" 和 "垫子 猫 坐 上 在"
  → Attention 无法区分！（因为所有 token 互相看到，没有顺序信息）

解决方案: 在输入中加入位置信息
```

### 5.2 正弦位置编码 vs 可学习位置编码 — 15min

```python
# 方法 1: Sinusoidal (原论文)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

# 方法 2: Learned (更常用，如 GPT、ViT)
self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
```

---

## 模块 6: 动手 — 手写 Mini Transformer（1.5 小时）

### 完整任务: 实现一个字符级语言模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========= 把上面所有的代码整理在一起 =========

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_O(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, block_size=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_model * 4) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)          # [B, T, d_model]
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        return self.lm_head(x)                       # [B, T, vocab_size]

# 用莎士比亚文本测试
# text = open('shakespeare.txt').read()
# ... (训练代码略，建议参考 Andrej Karpathy 的 nanoGPT)
```

**验证目标**: 模型能生成看起来像文本的字符序列（即使没有意义），说明 Transformer 在工作。

---

## 🎤 费曼挑战 + 论文精读（1.5 小时）

### 论文精读: "Attention Is All You Need" (30min)

**阅读顺序**（不要从头到尾读！）:

1. **Abstract** (5min): 读懂核心贡献
2. **Figure 1 + Figure 2** (10min): 对着架构图理解数据流
3. **Section 3.2: Attention** (10min): 逐公式对应代码
4. **Section 3.1: Scaled Dot-Product Attention** (5min): 理解 scale 的原因

**重点公式**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 费曼任务 1: 讲解 Transformer 给"昨天的自己" (30min)

录一段 10 分钟的视频/音频，向 Day 1 的自己解释:

1. Attention 解决了什么问题？(RNN → Attention)
2. Q, K, V 分别是什么？用找钥匙的类比解释
3. Multi-Head 为什么比单头好？
4. 残差连接 + LayerNorm 放在哪里？为什么？

**讲不清楚的地方 = 需要回去学的地方**

### 费曼任务 2: 画图挑战 (15min)

在一张 A4 纸上，凭记忆画出 Transformer 的完整架构图，标注每个模块的输入输出 shape。

### 费曼任务 3: 代码审视 (15min)

打开你写的 MiniTransformer，逐行解释每个变量 shape 的变化。不能跳过任何一行！

---

## 📝 今日自检清单

- [ ] 我能手算一个简单的 Self-Attention
- [ ] 我能画 Transformer 架构图（凭记忆）
- [ ] 我能解释 Q/K/V 的来源和作用
- [ ] 我知道为什么要除以 √d_k
- [ ] 我知道 Multi-Head 怎么并行计算
- [ ] 我能解释残差连接 + LayerNorm 的位置
- [ ] 我手写了 Mini Transformer 并跑通了
- [ ] 我读完了 "Attention Is All You Need" 的关键章节
- [ ] 我录制了费曼讲解

---

## 💡 理解 Checkpoint

| 概念 | 一句话解释 |
|------|-----------|
| Self-Attention | 让每个 token 看到所有 token，算加权和 |
| Multi-Head | 多组 QKV 并行，学不同的"关注模式" |
| Position Encoding | 告诉模型 token 的位置（否则 Attention 不知道顺序）|
| LayerNorm | 归一化每个样本的特征维度，稳定训练 |
| Residual | x + F(x)，给梯度一条"高速公路" |
| GELU | Transformer 的标配激活函数（更平滑的 ReLU） |

---

> [!info] 知识库关联
> - [[../../../01-基础理论/Transformer与注意力/Transformer与注意力|Transformer与注意力]] — 核心公式 + Multi-Head + 完整实现
> - [[../../../01-基础理论/Transformer与注意力/位置编码与归一化|位置编码与归一化]] — Sinusoidal/Learnable/RoPE + Pre-norm/Post-norm
> - [[../../../01-基础理论/Transformer与注意力/图解Transformer|图解Transformer]] — 可视化图解
> - [[../../../01-基础理论/Transformer与注意力/Cross-Attention与VLA应用|Cross-Attention与VLA应用]] — 模态融合 + VLA架构
>
> ✅ **完成打卡**: 今天的内容是 VLA 的根基，必须 100% 掌握！
>
> 🔜 **明日预告**: NLP 与 LLM — 理解 GPT 怎么生成文本，LoRA 怎么高效微调！
