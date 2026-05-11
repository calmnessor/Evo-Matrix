# π₀-FAST: Efficient Action Tokenization for Vision-Language-Action Models

> 用 DCT + BPE 将动作序列压缩 10×，训练加速 5×，效果不降。

## 论文信息

- **标题**: FAST: Efficient Action Tokenization for Vision-Language-Action Models
- **日期**: 2025 年 1 月
- **arXiv**: [2501.09747](https://arxiv.org/abs/2501.09747)
- **机构**: Physical Intelligence, UC Berkeley, Stanford
- **项目页**: https://pi.website/research/fast
- **开源**: HuggingFace `physical-intelligence/fast`

## 一句话总结

FAST 用 **DCT (离散余弦变换) + BPE (字节对编码)** 将高频动作序列 (~50Hz = 每秒 50+ 个动作 token) 压缩到 30-60 个 token，解决了 VLA 中动作 token 化效率的根本瓶颈，使自回归 VLA 训练加速 5×。

## 核心动机

### 动作 Token 化的困境

标准做法 (OpenVLA, RT-2)：
```
动作 [Δx, Δy, Δz, Δr, Δp, Δy, gripper] × 50 步/秒
→ 每维 256 bin → 每步 7 个离散 token
→ 1 秒 = 350 个动作 token
→ 加上视觉+文本 token → LLM 上下文爆炸
```

更糟的是：**相邻时刻的动作几乎一样** → token 序列高度冗余 → 自回归训练几乎没有信息增益。

### FAST 的洞察

借鉴 JPEG 压缩的思想：
1. **空间冗余** → DCT 变换（JPEG 的核心）
2. **统计冗余** → BPE 压缩（NLP 的核心）

```
原始动作序列 [100 timesteps, 7 dims] = 700 个 token
  ↓ DCT → 频域 + 截断高频系数
稀疏系数矩阵 [~30 values]
  ↓ BPE → 合并常见子序列
压缩 token [30-60 tokens]
```

**10× 压缩！**

## 方法详解

### 步骤 1: DCT (离散余弦变换)

将时域动作序列变换到频域：

```python
import numpy as np
from scipy.fft import dct

# action_chunk: [T, act_dim]  如 [100, 7]
def encode_dct(action_chunk):
    # 逐维度做 DCT (沿时间轴)
    freq_domain = dct(action_chunk, axis=0, norm='ortho')
    # 保留低频系数 (前 K 个)，截断高频
    freq_domain = freq_domain[:K]  # K << T, 如 K=8
    return freq_domain  # [K, act_dim]

def decode_dct(freq_domain, T):
    # 补零 + 逆 DCT
    padded = np.pad(freq_domain, ((0, T - len(freq_domain)), (0, 0)))
    return idct(padded, axis=0, norm='ortho')
```

**直觉**：
- 低频系数 = 动作的整体趋势 ("向右移动")
- 高频系数 = 动作的快速抖动 ("微小震荡")
- 截断高频 → 丢弃噪声，保留本质 → 动作反而更平滑

### 步骤 2: 量化

DCT 系数是连续值 → 离散化为整数：

```python
quantized = np.round(freq_domain / quantization_step)
```

### 步骤 3: BPE (字节对编码)

将离散化的系数矩阵展平为整数序列，用 BPE 合并常见子序列：

```
系数序列: [12, 3, 45, 12, 3, 45, 78, ...]
  ↓ BPE 训练
词表: {..., (12,3,45) → token_234, ...}
  ↓ 编码
压缩序列: [..., token_234, token_234, ...]
```

### FAST+ : 通用 Tokenizer

在 **1M+ 真实机器人动作轨迹**上训练，覆盖单臂/双臂/移动平台：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("physical-intelligence/fast")
tokens = tokenizer.encode(action_chunk)  # 30-60 tokens
actions = tokenizer.decode(tokens)        # 恢复动作
```

## 架构：π₀-FAST

```
图像 → VLM (PaliGemma) → 视觉 token
文本 → Tokenizer → 文本 token
动作 → FAST Tokenizer → 动作 token (30-60 个!)
           ↓
    [视觉 | 文本 | 动作] → Transformer Decoder → 自回归预测
```

**与 π₀ 的关系**：
- π₀: Flow Matching 生成连续动作 (训练慢但推理快)
- π₀-FAST: 自回归生成离散动作 token (训练快 5×，推理类似)

## 关键结果

### 训练效率

| 模型 | 训练时间 (相同数据) | 动作 token 数/秒 |
|------|------------------|---------------|
| π₀ (Flow Matching) | 基准 | 连续 (无需 token 化) |
| 标准离散 tokenization | ~2× 慢 | 350 |
| **π₀-FAST** | **0.2× (5× 快)** | **30-60** |

### 任务性能

π₀-FAST 在灵巧操作任务上匹配或超越 π₀：
- 叠衣服：相当
- 清理桌面：相当
- 打包购物袋：略优

### 零样本泛化

首个在 **DROID 全数据集**上实现零样本泛化的模型——跨不同大学校园未见过的环境。

## 优势与局限

**优势**：
- 训练快 5× → 迭代速度大幅提升
- 动作 token 数减少 10× → LLM 上下文压力大降
- FAST+ 通用 tokenizer 开箱即用
- 自回归架构与 LLM 训练 pipeline 天然兼容

**局限**：
- 自回归推理较慢 (~750ms/秒动作) vs Flow Matching (~100ms/秒)
- DCT 截断可能丢失精细的高频动作细节
- 量化+压缩有信息损失（实践中很小）

## 自检问题

- [ ] 我理解 DCT 如何将动作从时域变换到频域
- [ ] 我知道为什么截断高频系数反而让动作更平滑
- [ ] 我能解释 FAST 的 10× 压缩来自哪里（DCT 压缩 + BPE 压缩）
- [ ] 我理解 π₀-FAST (自回归离散) 和 π₀ (Flow Matching 连续) 的优劣

## 关联笔记

- [[π₀]] — Flow Matching 版本，FAST 的对比基线
- [[../π₀.₅-KI]] — 知识绝缘版中 FAST 的角色
- [[../../../VLA模型总览]] — 动作 token 化的三种方案对比
- [[../../../OpenX-Embodiment数据集]] — FAST 训练所用的部分数据来源
