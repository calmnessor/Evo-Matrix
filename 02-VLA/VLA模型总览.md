# VLA 模型总览

> 从 RT-1 到 OpenVLA 到 π₀——VLA 模型家族谱系、架构对比、技术演进全景。

## 发展时间线

```
2022: RT-1 — 首次大规模 Transformer 用于机器人控制
2023: RT-2 — 用 VLM 输出动作 token (Vision-Language-Action 概念诞生)
2023: ACT — 动作分块 + 条件 VAE (CVAE 用于动作生成)
2023: Diffusion Policy — 扩散模型用于动作生成
2023: RT-X (OXE) — 跨机器人数据集 + 跨 embodiment 泛化
2024: Octo — 第一个通用机器人 Transformer 基座模型
2024: OpenVLA — 开源 7B VLA, SigLIP + LLaMA + LoRA
2024: π₀ (Physical Intelligence) — Flow Matching, 大规模通用操作
2024: GR00T (NVIDIA) — 工业级多模态基础模型
2024: RDT (清华) — 扩散 Transformer + 双臂操作
2025: 更大规模、跨 embodiment、灵巧操作、Sim-to-Real
```

## 三大技术路线

### 路线 1: LLM + 动作 Token 化 (RT-2 风格)

```
摄像头图像 → ViT/SigLIP → 视觉 token
文本指令 → Tokenizer → 文本 token
                              ↓
                    LLM 自回归生成 → 动作 token
                              ↓
                    De-tokenize → 连续动作
```

**代表**: RT-2, OpenVLA, RoboFlamingo, LLaRA

**优势**: 利用 LLM 常识推理（从互联网文本学到物理世界知识），多任务泛化强，prompt 可控

**劣势**: 推理慢 (~500ms for 7B)，离散化有精度损失，有限的动作 token bin 数

**核心创新点**：把机器人控制转化为语言建模 —— "生成下一个 token"，只是这个 token 恰好是动作。

### 路线 2: 扩散策略 (Diffusion Policy 风格)

```
观测 + 噪声动作 → Denoising U-Net/Transformer → 预测噪声 → 逐步去噪 → 动作序列
```

**代表**: Diffusion Policy, RDT, DICE

**优势**: 生成平滑轨迹，天然多模态分布（左绕/右绕都能表达），推理快（轻量网络）

**劣势**: 多步去噪需要反复前向传播（DDIM 加速到 ~10 步缓解），不天然利用语言

**核心创新点**：动作空间的"形象"由噪声→去噪的过程塑造，天然支持多峰动作分布。

### 路线 3: 条件 VAE 生成 (ACT 风格)

```
观测 → Encoder → latent z（采样） → Decoder → 动作序列 (Action Chunk)
```

**代表**: ACT, Droid, ALOHA 系列

**优势**: 一次推理出序列动作（Chunking），推理极快，实现简单

**劣势**: VAE prior 可能欠拟合 → 某些动作模式生成不出来

**核心创新点**：Action Chunking + Temporal Ensemble 大幅降低有效推理频率。

### 路线 4 (新兴): Flow Matching (π₀ 风格)

```
观测 + 噪声动作 → 预测向量场 → 沿向量场积分 → 动作序列
```

**代表**: π₀, RDT-2

**优势**: 比扩散更灵活的训练目标（不限于预测噪声），推理路径更短

**劣势**: 较新，工程成熟度不如扩散

## 模型详细对比

| 模型 | 年份 | 机构 | Backbone | 动作表示 | 动作维度 | 推理频率 | 开源 | 训练数据 |
|------|------|------|---------|---------|---------|---------|------|---------|
| **RT-1** | 2022 | Google | EffNet + 小 Transf. | 连续 | 7 | ~35Hz | ❌ | 130K 轨迹 |
| **RT-2** | 2023 | Google | PaLI-X / PaLM-E | 离散 token (256 bins) | 7 | ~2Hz | ❌ | OXE + Web |
| **ACT** | 2023 | Stanford | ResNet + Transf. | 连续 + chunk (16) | 7 | ~50Hz* | ✅ | 数百条 |
| **Diffusion Policy** | 2023 | Columbia | CNN/Transf. | 扩散去噪 + chunk | 7-14 | ~10Hz* | ✅ | 数百条 |
| **Octo** | 2024 | Berkeley | Transformer | 扩散 | 7 | ~10Hz | ✅ | OXE (800K) |
| **OpenVLA** | 2024 | Stanford | SigLIP + LLaMA-7B | 离散 token (256 bins) | 7 | ~3Hz | ✅ | OXE (970K) |
| **π₀** | 2024 | Physical Int. | 自研 VLM | Flow Matching | 可变 | ~50Hz | ❌ | 10K 小时 |
| **GR00T** | 2024 | NVIDIA | 多模态基座 | 多种 | 可变 | N/A | ❌ | 大规模 |
| **RDT** | 2024 | 清华 | DiT (扩散 Transf.) | 扩散 | 14-28 | ~10Hz* | ✅ | 多源混合 |

> *ACT 一次推理 16 步取前 8 步 → 等效推理频率远低于执行频率。Diffusion Policy 类似。

## 代表性架构深度拆解

### RT-1 Architecture

```
图像序列 (6 帧 @ 3Hz = 2 秒历史)
    ↓
EfficientNet-B3 (每帧独立 → 然后 concat)
    ↓
FiLM (语言指令注入: γ, β 乘/加到视觉特征上)
    ↓
TokenLearner (从 486 个 spatial token 压缩到 8 个)
    ↓
Transformer Encoder (8 个 token + 位置编码)
    ↓
MLP Head → [dx, dy, dz, dr, dp, dy, gripper_open]
```

**TokenLearner** 是 RT-1 最巧妙的创新：用 8 个可学习 query 对 486 个 spatial feature 做 soft-attention，选出最重要的 8 个位置。参数量极省，却比平均池化好得多。

### RT-2 Architecture

```
图像 → ViT (PaLI-X 的视觉分支) → [N, 1152] visual tokens
文本 → Tokenizer → [L] text tokens
              ↓
    PaLI-X / PaLM-E Decoder 自回归生成:
    
    生成序列: [... text response ...] [ACT_START] [Δx_bin] [Δy_bin] ... [gripper_bin] [ACT_END]
              ↑ 模型可以先用语言"思考"再输出动作        ↑ 每个维度离散化为 256 个 bin

De-tokenize: bin_index → 对应 bin 的中心值 → 连续动作
```

**革命性创新**：动作 token 和文本 token 使用同一个词表。对 LLM 来说，生成"向前移动 5cm"和生成"Hello World"没有本质区别——都是预测下一个 token。这让 LLM 的预训练知识可以直接迁移到机器人控制。

### OpenVLA Architecture

```
输入: 单帧 RGB [384, 384, 3] + 语言指令

视觉: SigLIP SO400M (patch=14, 27 layers, 1152 hidden)
    → 729 visual tokens [729, 1152]

投影器: 2 层 MLP (1152 → 4096, 匹配 LLaMA hidden dim)
    → [729, 4096]

语言: LLaMA Tokenizer → [L] text tokens
    → LLaMA Embedding → [L, 4096]

拼接: [729, 4096] + [L, 4096] = [729+L, 4096]
    → LLaMA-7B Decoder (32 layers, 32 heads, d_k=128)
    → 取最后一个 token 的 hidden state → 动作预测

动作: 7 个独立分类头 (每个维度 256 bins)
    → Cross-Entropy Loss (因离散化了)
    → 推理: argmax → bin center → 连续值

微调: LoRA (r=32) on Q, V projections
    → 只需训练 ~20M 参数 (不到全量的 0.3%)
    → 每个新任务只需 ~40MB 的 LoRA 权重
```

### 动作空间对比

| 模型 | 动作表示 | Loss | 优点 | 缺点 |
|------|---------|------|------|------|
| RT-1 | 连续 [dx,...,gripper] | MSE | 简单、无精度损失 | 单峰 |
| RT-2 | 离散 256 bins × 7 dims | Cross-Entropy | 融入 LLM 自然 | 精度上限 (bin size) |
| OpenVLA | 离散 256 bins × 7 dims | Cross-Entropy | 同上 | 同上 |
| ACT | 连续 + chunk 16 步 | MSE + β KL | 快、平滑 | VAE prior |
| DiffPolicy | 连续 + chunk + 噪声预测 | MSE(noise) | 多模态 | 推理慢 |
| π₀ | 连续 + Flow Matching | 向量场匹配 | 灵活 | 新方法 |

## 训练范式演进

```
2022-2023: 单任务/单机器人 BC
  RT-1, ACT, Diffusion Policy → 数百到数万条轨迹

2023-2024: 多任务/多机器人 联合 BC
  RT-X, Octo, OpenVLA → OXE 百万级轨迹 → 跨 embodiment 泛化

2024: BC 预训练 + RL 微调
  RLPD, SERL → BC 提供好起点, RL 在仿真中精调

2025+: 更大规模 + 自监督
  利用人类视频、3D 重建、世界模型 → 突破机器人数据瓶颈
```

## 选型建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 入门学习 | ACT / Diffusion Policy | 代码简单、轻量、能跑自己的数据 |
| 科研 benchmark | OpenVLA | 开源、有社区、可 LoRA 微调 |
| 多任务泛化 | Octo / OpenVLA | OXE 预训练给了强泛化基础 |
| 精细灵巧操作 | Diffusion Policy / π₀ | 扩散的平滑性和 Flow Matching 的灵活性 |
| 工业部署 | GR00T / 自研 | 闭源但性能最强、有 NVIDIA 生态 |

## 推荐阅读

| # | 论文/资源 | 关键词 |
|---|----------|--------|
| 1 | **RT-2** (2307.15818) | 动作 token 化，VLA 范式诞生 |
| 2 | **OpenVLA** (2406.09246) | 开源 VLA 完整方案 |
| 3 | **Diffusion Policy** (2303.04137) | 扩散动作生成 |
| 4 | **ACT** (2304.13705) | CVAE + Action Chunking |
| 5 | **RT-X** (2310.08864) | 跨 embodiment 泛化 |
| 6 | **Octo** (2405.12213) | 通用机器人基座模型 |
| 7 | **π₀** (Physical Intelligence blog) | Flow Matching VLA |

## 自检问题

### 基础关
- [ ] 我能画出 RT-1 和 RT-2 的架构对比图
- [ ] 我理解 Action Tokenization 的原理（连续动作 → 离散 bin → token）
- [ ] 我能说出三大（+Flow Matching 的四大）技术路线的优劣
- [ ] 我知道 OpenVLA 用的视觉 (SigLIP) 和语言 (LLaMA) backbone 及参数量
- [ ] 我能列举 VLA 发展时间线上的关键节点

### 进阶关
- [ ] 我理解 RT-1 的 TokenLearner 如何压缩 token
- [ ] 我能解释 OpenVLA 中 LoRA 微调的具体配置 (r, α, target_modules)
- [ ] 我能对比离散动作 token (RT-2/OpenVLA) 和连续动作 (ACT) 的优劣
- [ ] 我知道 OXE 共训练带来的泛化提升及其理论解释
- [ ] 我理解 Flow Matching 和 DDPM 在动作生成上的区别

### 实战关
- [ ] 我能跑通 OpenVLA 的推理
- [ ] 我理解 OpenVLA 源码中的模型定义
- [ ] 我做过至少一个模型的 LoRA 微调
- [ ] 我能根据场景选择合适的 VLA 方案并给出理由

## 关联笔记
- [[Transformer与注意力]] — 所有 VLA 的核心引擎
- [[ViT/ViT概述|ViT概述]] — 视觉编码器对比 (SigLIP/CLIP/DINOv2)
- [[大语言模型与微调]] — LLM backbone + LoRA
- [[行为克隆]] — VLA 的主要训练方式
- [[扩散策略]] — 路线 2 的详细展开
- [[ACT]] — 路线 3 的详细展开
- [[多模态融合架构]] — 视觉+语言 token 如何融合
- [[OpenX-Embodiment数据集]] — 数据基础
- [[机器人运动学]] — 动作空间的物理含义
