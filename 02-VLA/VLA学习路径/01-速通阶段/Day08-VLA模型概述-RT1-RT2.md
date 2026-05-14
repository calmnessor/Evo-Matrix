# 🔥 Day 8: VLA 模型概述 — RT-1 与 RT-2（8 小时）

> **口号**: "直击 VLA 核心——从 RT-1 到 RT-2，见证范式的诞生！"  
> **目标**: 完整理解 RT-1 和 RT-2 的架构、训练方式和核心创新  
> **为什么重要**: RT-1/RT-2 定义了 VLA 这个方向，后续所有模型（OpenVLA 等）都基于此

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | RT-1: Robotics Transformer | 2h | [ ] |
| 2 | RT-2: Vision-Language-Action | 2h | [ ] |
| 3 | RT-1 vs RT-2 对比分析 | 1h | [ ] |
| 4 | Action Tokenization 深入 | 1h | [ ] |
| 5 | 论文精读 | 1h | [ ] |
| 6 | 费曼挑战 | 1h | [ ] |

---

## 模块 1: RT-1 — Robotics Transformer（2 小时）

### 1.1 历史地位 — 15min

```
论文: "RT-1: Robotics Transformer for Real-World Control at Scale"
机构: Google Robotics (2022)

核心贡献:
  1. 首次将 Transformer 大规模用于真实机器人控制
  2. 提出了将图像历史和指令共同编码的架构
  3. 证明了大规模多任务训练的有效性
  4. 在 700+ 任务上训练，展示了强大的泛化能力

一句话: RT-1 证明了 "Transformer + 大规模机器人数据" 这条路能走通
```

### 1.2 RT-1 架构详解 — 50min

```
RT-1 架构（从输入到输出）:

输入:
  1. 6 帧历史图像 [6, H, W, 3]
  2. 自然语言指令 (如 "pick up the red can")

处理流程:

步骤 1: 图像 → Token
  每帧经过 ImageNet 预训练的 EfficientNet → 9×9×512 的特征图
  6 帧共 6 × 81 = 486 个图像 token

步骤 2: 指令 → Token  
  通过 Universal Sentence Encoder → 语言 embedding
  复制成 486 个 token（和图像 token 一一对应）

步骤 3: FiLM 条件注入
  FiLM = Feature-wise Linear Modulation
  用语言特征对图像特征做仿射变换:
    FiLM(feat, lang) = γ(lang) * feat + β(lang)

步骤 4: TokenLearner
  从 486 个 token 中选出最重要的 8 个 token
  → 压缩信息，减少后续 Transformer 的计算量

步骤 5: Transformer
  8 个 token → Transformer Encoder × 8 layers
  → 输出 8 个增强的 token

步骤 6: Action Head
  将 Transformer 输出映射为动作:
    [dx, dy, dz, droll, dpitch, dyaw, gripper_open, base_movement]
```

```python
# RT-1 的伪代码实现（概念级）
class RT1(nn.Module):
    def __init__(self):
        super().__init__()
        # 视觉编码
        self.visual_encoder = EfficientNetB3(pretrained=True)

        # TokenLearner: 把 N 个 token 压缩为 M 个（M << N）
        self.token_learner = TokenLearner(num_output_tokens=8)

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=8,
        )

        # 动作输出
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7 + 3),  # 7 DOF arm + 3 DOF base
        )

    def forward(self, images, instruction_embedding):
        B, T, C, H, W = images.shape  # T=6

        # 1. 每帧提取特征
        img_tokens = []
        for t in range(T):
            feat = self.visual_encoder(images[:, t])  # [B, 9, 9, 512]
            img_tokens.append(feat.flatten(1, 2))     # [B, 81, 512]
        img_tokens = torch.cat(img_tokens, dim=1)     # [B, 486, 512]

        # 2. FiLM 注入语言信息
        lang_gamma = self.lang_to_gamma(instruction_embedding)
        lang_beta = self.lang_to_beta(instruction_embedding)
        img_tokens = lang_gamma * img_tokens + lang_beta

        # 3. TokenLearner 压缩
        tokens = self.token_learner(img_tokens)  # [B, 8, 512]

        # 4. Transformer
        tokens = self.transformer(tokens)        # [B, 8, 512]

        # 5. 输出动作
        action = self.action_head(tokens[:, 0])  # [B, action_dim]
        return action


# TokenLearner 的核心（简化）
class TokenLearner(nn.Module):
    def __init__(self, num_output_tokens=8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, num_output_tokens),
        )

    def forward(self, x):
        """x: [B, N, D] → [B, M, D]"""
        # 每个 token 学一个"重要性"分数
        weights = self.attention(x)  # [B, N, M]
        weights = F.softmax(weights, dim=1)

        # 加权聚合
        return weights.transpose(1, 2) @ x  # [B, M, D]
```

### 1.3 训练细节 — 20min

```
训练数据:
  13 台机器人，17 个月
  700+ 任务
  130,000+ episodes

训练方式:
  BC (行为克隆)
  MSE Loss on actions
  Batch size: 256
  Sequence length: 6 帧

关键发现:
  1. 多任务训练 → 比单任务好 3 倍
  2. 大规模数据 → 显著提升泛化
  3. 图像增强 → 鲁棒性来源
```

### 1.4 RT-1 的局限 — 10min

```
1. 不是端到端的 VLM
   - 视觉编码器是固定的 (EfficientNet)
   - 语言编码器是固定的 (USE)
   - 没有利用 LLM 的推理能力

2. 不能结合语言和视觉推理
   - "拿起左边的红色罐子"需要理解"左"、"红色"
   - RT-1 只是条件注入，没有真正的语义理解

3. 动作空间固定
   - 不能泛化到新机器人

→ 这些局限推动了 RT-2 的诞生
```

---

## 模块 2: RT-2 — Vision-Language-Action（2 小时）

### 2.1 核心创新 — 20min

```
论文: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
机构: Google DeepMind (2023)

革命性想法:
  把机器人动作也当成"语言 token"！

传统方式 (RT-1):
  图像 + 文本 → 理解 → MLP 头 → 连续动作值

RT-2 方式:
  图像 + 文本 → 大语言模型  → 自回归生成 token → 动作

为什么革命性?
  - 可以利用 LLM 从互联网学到的知识（常识推理）
  - 可以利用 VLM 的视觉理解能力
  - 统一了视觉、语言、动作的表示
```

### 2.2 RT-2 架构详解 — 50min

```
RT-2 的整体架构:

      图像                    文本指令
       │                       │
  ┌────▼────┐            ┌─────▼──────┐
  │  ViT    │            │ Tokenizer  │
  │ (PaLI-X │            │ (PaLM-E    │
  │  style) │            │  tokenizer)│
  └────┬────┘            └─────┬──────┘
       │                       │
       │   视觉 token           │   文本 token
       │   [N_img tokens]      │   [N_text tokens]
       │                       │
       └───────┬───────────────┘
               │
      ┌────────▼────────┐
      │   大语言模型     │
      │   (PaLI-X 或    │
      │    PaLM-E)      │
      │                 │
      │  自回归生成:    │
      │  → 文本 token   │
      │  → 动作 token   │
      │  → 文本 token   │
      └────────┬────────┘
               │
      ┌────────▼────────┐
      │  Token → Action │
      │  De-tokenizer   │
      └────────┬────────┘
               │
          动作指令
    [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

### 2.3 Action Tokenization — 动作如何变成 token？ — 30min

```python
# RT-2 的核心创新: 把连续动作离散化

def action_to_tokens(action_vector, num_bins=256):
    """
    连续动作 → 离散 token 序列

    action_vector: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    每个维度归一化到 [-1, 1]
    """
    tokens = []

    for dim_value in action_vector:
        # 把 [-1, 1] 均匀分成 256 个 bin
        bin_index = int((dim_value + 1) / 2 * (num_bins - 1))
        bin_index = max(0, min(bin_index, num_bins - 1))

        # 这个 bin_index 就对应一个特殊的 token
        # 例如: dim0_bin_128, dim1_bin_50, ...
        token_id = ACTION_VOCAB_START + bin_index
        tokens.append(token_id)

    return tokens

# 例子:
# action = [0.5, -0.3, 0.1, 0, 0, 0, 1.0]
# tokens = [ACT_0_192, ACT_1_90, ACT_2_141, ACT_3_128, ACT_4_128, ACT_5_128, ACT_6_255]

# 这样 LLM 就可以像生成文字一样生成动作！
# 模型输出 "拿起 [ACT_0_192] [ACT_1_90] ... 物体" 的形式
```

### 2.4 两种 RT-2 变体 — 15min

```
RT-2-PaLI-X (视觉语言模型路线):
  基础模型: PaLI-X (55B 参数的 VLM)
  输入: 图像 + 文本 → 输出: text tokens + action tokens
  优势: 原生多模态理解

RT-2-PaLM-E (语言模型 + 视觉注入路线):
  基础模型: PaLM-E (将视觉特征注入 LLM)
  输入: 视觉特征 + 文本 → 输出: text tokens + action tokens
  优势: 更强的推理能力 (用了更大的 LLM)

共同点:
  - 都用 LLM/VLM backbone
  - 都做 action tokenization
  - 都用 BC 训练
```

### 2.5 关键实验结果 — 10min

```
1. 涌现能力 (Emergent Capabilities):
   - 理解符号（"把可乐罐移到 Taylor Swift 照片旁"）
   - 推理（"把苹果移到原来放可乐的地方"）
   - 人类识别（"把饮料给看起来累的人"）

2. 泛化能力:
   RT-2 在未见过的新任务上显著优于 RT-1
   互联网知识 → 机器人能力的转移成功了！

3. 多语言:
   用英语训练，能理解其他语言的指令
```

---

## 模块 3: RT-1 vs RT-2 对比分析（1 小时）

### 3.1 架构对比 — 30min

```
                RT-1                    RT-2
┌───────────┬─────────────────┬─────────────────────┐
│ 视觉编码   │ EfficientNet    │ ViT (CLIP/VLM backbone)│
│ 语言编码   │ USE (固定)       │ LLM tokenizer + embedding│
│ 核心模型   │ 小 Transformer  │ 大 VLM/LLM (55B+)    │
│ 动作表示   │ 连续值 (MLP头)   │ 离散 token (自回归生成)│
│ 训练数据   │ 130K episodes   │ 互联网图文 + 机器人数据│
│ 推理能力   │ 弱               │ 强（LLM带来的）       │
│ 模型大小   │ ~50M             │ 55B (千倍差距！)      │
│ 时序建模   │ 6帧历史          │ 无显式时序            │
└───────────┴─────────────────┴─────────────────────┘
```

### 3.2 关键设计抉择 — 30min

```
抉择 1: 连续 vs 离散动作

RT-1 (连续):
  优点: 精度高，直接回归
  缺点: 不能利用 LLM，多模态困难

RT-2 (离散):
  优点: 融入 LLM 生态，统一 token 空间
  缺点: 量化误差 (256 bins 的精度有限)

抉择 2: 小模型 vs 大模型

RT-1 小模型:
  优点: 推理快 (~50ms)，可以高频率控制
  缺点: 泛化弱

RT-2 大模型:
  优点: 泛化强，有常识推理
  缺点: 推理慢 (~500ms)，需要 Action Chunking 补偿
```

---

## 模块 4: Action Tokenization 深入（1 小时）

### 4.1 离散步长的选择 — 20min

```python
def analyze_action_discretization(num_bins=256, action_range=(-1, 1)):
    """分析离散化精度"""
    bin_width = (action_range[1] - action_range[0]) / num_bins
    max_error = bin_width / 2

    print(f"Bins: {num_bins}")
    print(f"Bin width: {bin_width:.6f}")
    print(f"Max quantization error: {max_error:.6f}")

    # 对于位置控制 (假设 ±1 对应 ±10cm)
    max_position_error_mm = max_error * 100
    print(f"Max position error (if ±10cm): {max_position_error_mm:.2f} mm")

    return bin_width, max_error

# bin_width=256 时: 每个 bin 约 0.0078 → 精度约 0.78mm，足够

# 但是！这个精度对于精细操作（如穿针）可能不够
# 解决方案: 
#   1. 增加 bin 数 → 但 token 表会变大
#   2. 混合表示: 粗粒度 bin + 细粒度残差
#   3. 多尺度量化
```

### 4.2 现代 Action Tokenization 方法 — 25min

```python
# 方法 1: RT-2 风格（均匀量化）
def rt2_style(action, num_bins=256):
    return [(a + 1) / 2 * (num_bins - 1) for a in action]

# 方法 2: 矢量量化 (VQ-VAE 风格)
class VQActionTokenizer(nn.Module):
    """把动作序列压缩到离散 codebook"""
    def __init__(self, action_dim, codebook_size=1024, latent_dim=64):
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, actions):
        z = self.encoder(actions)
        # 找最近的 codebook vector
        distances = torch.cdist(z, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        quantized = self.codebook(indices)
        reconstructed = self.decoder(quantized)
        return reconstructed, indices, z

# 方法 3: GPT 风格（逐 token 预测）
# 把 action 序列当成"动作语言"
# [ACT_START] [dim0_b192] [dim1_b90] [dim2_b141] ... [ACT_END]
# 然后 LLM 自回归生成
```

### 4.3 对 VLA 推理性能的影响 — 15min

```
推理速度分析:
  
  RT-1 (连续回归):
    1 次 forward → 7 个连续值
    延迟: ~50ms → 可以 20Hz 控制

  RT-2 (自回归生成):
    需要生成 7 个 action token → 7 次 forward
    延迟: ~500ms → 只能 2Hz 控制
    + KV cache 可以加速，但仍然慢

  解决方案:
    → Action Chunking: 一次生成 K 组动作 (如 10 组 = 70 个 token)
    → 执行 K 步，同时生成下一批
    → 或者用 Draft Model (投机解码)
```

---

## 模块 5: 论文精读（1 小时）

### RT-1 论文 — 30min

**必读章节**:
1. Section 2: Related Work (快速过，了解背景)
2. Section 3: Model Architecture (精读，对照上面的架构图)
3. Section 4: Training (了解数据规模和训练方式)
4. Section 5: Experiments (重点看图表)

**核心问题** （读完后回答）:
1. TokenLearner 压缩比是多少？（486 → 8）
2. 为什么用 FiLM 而不是拼接做条件注入？
3. RT-1 的数据增强有哪些？为什么重要？

### RT-2 论文 — 30min

**必读章节**:
1. Introduction (完整读！写的非常好)
2. Section 3: RT-2 Framework (精读)
3. Section 4: Action Tokenization (核心章节)
4. Section 5: Emergent Capabilities (最精彩的部分)

**核心问题** （读完后回答）:
1. "拿起 Taylor Swift 照片旁的可乐罐" 为什么 RT-1 做不到而 RT-2 能做到？
2. RT-2 的 co-fine-tuning 是怎么做的？为什么重要？

---

## 🎤 费曼挑战（1 小时）

### 任务 1: 架构默写 (20min)
不参考任何资料，默画出:
- RT-1 的完整数据流图（标注每个模块的输入输出 shape）
- RT-2 的完整数据流图（标注 action tokenization 的位置）

### 任务 2: 技术演进讲解 (25min)
假设你是要写一本《VLA 发展史》的作者，请口头讲解:
1. RT-1 解决了什么以前没解决的问题？
2. RT-2 为什么是革命性的？（从 action tokenization 角度）
3. 如果让你设计 RT-3，你会改进什么？

### 任务 3: 面试题 (15min)
**"RT-2 如何将互联网知识迁移到机器人控制？请从架构和训练角度解释"**

写出 200 字的回答。

---

## 📝 今日自检清单

- [ ] 我能画出 RT-1 的完整架构（从图像到动作）
- [ ] 我理解 TokenLearner 的原理
- [ ] 我理解 FiLM 条件注入
- [ ] 我能解释 RT-2 的 Action Tokenization
- [ ] 我知道 RT-2 两种变体的区别
- [ ] 我能对比 RT-1 和 RT-2 的核心差异
- [ ] 我知道为什么要把连续动作离散化
- [ ] 我读完了 RT-1 和 RT-2 论文的关键章节
- [ ] 我完成了费曼讲解

---

## 💡 核心概念速查

| 概念 | 解释 |
|------|------|
| TokenLearner | 通过学习的重要性权重，把 N 个 token 压缩成 M 个 |
| FiLM | 用语言特征对图像特征做缩放+平移 |
| Action Tokenization | 连续动作 → 离散 bin → token → LLM 生成 |
| Co-Fine-Tuning | 在机器人数据和互联网数据上联合微调 |
| Emergent Capability | 模型规模增大后自然出现的、未经专门训练的能力 |

---

> [!info] 知识库关联
> - [[../../VLA模型总览|VLA模型总览]] — 现有VLA模型全景对比
> - [[../../VLA方向综述|VLA方向综述]] — VLA领域发展脉络
> - [[../../论文精读/|VLA 论文精读目录]] — 22+ 篇已分析论文笔记
> - [[../../论文精读/pi系列/|π 系列总览]] — PI的完整技术栈 (π₀ → π₀.₆)
>
> ✅ **完成打卡**: VLA 两大基石已掌握！明天探索开源 VLA！
>
> 🔜 **明日预告**: Day 9 — OpenVLA 深入，部署并推理一个真正的开源 VLA 模型！
