# VLA 模型总览

> 从 RT-1 到 OpenVLA 到 π₀——VLA 模型家族谱系

## 发展时间线

```
2022: RT-1 — 首次大规模 Transformer 用于机器人控制
2023: RT-2 — 用 VLM 输出动作 token (Vision-Language-Action 概念诞生)
2023: ACT — 动作分块 + 条件 VAE
2023: Diffusion Policy — 扩散模型用于动作生成
2024: Octo — 通用机器人基座模型
2024: OpenVLA — 开源 VLA (7B 参数，LLaMA + SigLIP)
2024: π₀ (Physical Intelligence) — 大规模通用操作
2024: GR00T (NVIDIA) — 工业级基础模型
2025: 更大规模、更通用、sim-to-real 深化
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
**代表**: RT-2, OpenVLA
**优势**: 利用 LLM 常识推理，多任务泛化强
**劣势**: 推理慢 (~500ms)，离散化有精度损失

### 路线 2: 扩散策略 (Diffusion Policy 风格)
```
观测 + 噪声 → Denoising U-Net/Transformer → 去噪后动作序列
```
**代表**: Diffusion Policy
**优势**: 生成平滑轨迹，天然多模态分布，推理快
**劣势**: 训练需要多步去噪

### 路线 3: 条件 VAE 生成 (ACT 风格)
```
观测 → 编码器 → latent z → 解码器 → 动作序列
```
**代表**: ACT
**优势**: 生成动作块减少累积误差
**劣势**: VAE prior 可能欠拟合

## 模型对比

| 模型 | 年份 | Backbone | 动作表示 | 输入 | 开源 |
|------|------|---------|---------|------|------|
| RT-1 | 2022 | EfficientNet + 小 Transformer | 连续回归 | 6帧 + 指令 | ❌ |
| RT-2 | 2023 | PaLI-X / PaLM-E | 离散 token | 单帧 + 指令 | ❌ |
| Octo | 2024 | Transformer | 扩散 | 图像 + 指令 | ✅ |
| OpenVLA | 2024 | SigLIP + LLaMA | 离散 token | 单帧 + 指令 | ✅ |
| ACT | 2023 | CNN + Transformer | 连续 + chunk | 单帧 + 关节 | ✅ |
| DiffPolicy | 2023 | CNN/Transformer | 扩散 | 多帧观测 | ✅ |
| π₀ | 2024 | 自研 VLM | Flow Matching | 多模态 | ❌ |

## RT-1 架构

```
图像 (6帧) → EfficientNet → FiLM(语言注入) → TokenLearner(压缩到8)
→ Transformer Encoder → MLP → 连续动作 [dx, dy, dz, dr, dp, dy, gripper]
```

**核心创新**:
- TokenLearner: 486 → 8 个 token
- FiLM 条件注入（而非拼接）
- 大规模多任务 BC 训练

## RT-2 架构

```
图像 → ViT → 视觉 token
文本 → Tokenizer → 文本 token
          ↓
   LLM 自回归生成: [... text ...] [ACT_START] [dim0_bin] ... [dim7_bin] [ACT_END]
          ↓
  De-tokenize: bin_id → 连续值 → 动作
```

**革命性创新**: 动作 token 化 → 把机器人控制变成语言建模问题

## OpenVLA

- 视觉: SigLIP (google/siglip-so400m-patch14-384)
- 语言: LLaMA-7B
- 训练: OXE 数据 + LoRA 微调
- 完全开源，VLA 学习的主力平台

## 自检问题
- [ ] 我能画出 RT-1 和 RT-2 的架构对比图
- [ ] 我理解 Action Tokenization 的原理
- [ ] 我能说出三大技术路线的优劣
- [ ] 我知道 OpenVLA 用的视觉和语言 backbone

## 关联笔记
- [[Transformer与注意力]] — 所有 VLA 的核心引擎
- [[计算机视觉与ViT]] — 视觉编码器
- [[大语言模型与微调]] — LLM backbone
- [[行为克隆]] — VLA 的主要训练方式
- [[扩散策略]]
- [[ACT]]
- [[多模态融合架构]]
- [[OpenX-Embodiment数据集]]
