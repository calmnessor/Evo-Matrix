# 🎯 具身智能 VLA 学习路径

> **目标**: 从零基础到能独立阅读、部署、微调 VLA 模型，达到实习水平  
> **时间**: 第一阶段 14 天速通（每天 8 小时），第二阶段 30 天深入学习  
> **方法**: 费曼学习法（学 → 讲 → 简化 → 回顾）× 打卡制

---

## 📖 什么是 VLA？

**VLA (Vision-Language-Action)** 是具身智能的核心技术范式：让机器人**看**（Vision）、**理解指令**（Language）、**生成动作**（Action）。

```
摄像头图像 → [视觉编码器] → 视觉特征 ─┐
                                      ├→ [多模态融合] → [动作解码器] → 机器人关节指令
人类指令   → [语言编码器] → 语言特征 ─┘
```

### 核心模型谱系

| 模型 | 年份 | 核心贡献 | 关键词 | 笔记 |
|------|------|----------|--------|------|
| **RT-1** | 2022 | 将 Transformer 用于机器人控制 | Robotics Transformer | — |
| **RT-2** | 2023 | 直接用 VLM 输出动作 token | Vision-Language-Action | — |
| **ACT** | 2023 | 动作分块 + 条件 VAE | Action Chunking | [[../ACT\|ACT]] |
| **Diffusion Policy** | 2023 | 扩散模型用于动作生成 | Denoising Actions | [[../扩散策略\|扩散策略]] |
| **Octo** | 2024 | 通用机器人基座模型 | Generalist Robot Policy | — |
| **OpenVLA** | 2024 | 开源 VLA（7B 参数） | Open-Source VLA | [[../VLA模型总览\|VLA模型总览]] |
| **π₀ (PI-Zero)** | 2024 | 大规模通用策略 + Flow Matching | Generalist Manipulation | [[../论文精读/pi系列/\|π 系列]] |
| **GR00T** | 2024 | NVIDIA 通用机器人模型 | Foundation Model | — |
| **χ₀ (KAI-0)** | 2026 | 分布对齐 + 模型算术 | Robust VLA | [[../论文精读/框架与优化/KAI0/KAI0\|KAI0笔记]] |

---

## 🗺️ 知识体系全景图

```
                              ┌─────────────────────┐
                              │    VLA 算法大师      │
                              └──────────┬──────────┘
           ┌─────────────────────────────┼─────────────────────────────┐
           │                             │                             │
    ┌──────▼──────┐              ┌──────▼──────┐              ┌──────▼──────┐
    │  视觉理解    │              │  语言理解    │              │  动作生成    │
    │  Vision     │              │  Language   │              │  Action     │
    └──────┬──────┘              └──────┬──────┘              └──────┬──────┘
           │                             │                             │
    ┌──────▼──────┐              ┌──────▼──────┐              ┌──────▼──────┐
    │ ViT/ResNet  │              │ Transformer │              │ BC/IL/RL    │
    │ 3D Vision   │              │ LLM/LoRA    │              │ Diffusion   │
    │ Point Cloud │              │ Tokenizer   │              │ Autoreg.    │
    └─────────────┘              └─────────────┘              └─────────────┘
           │                             │                             │
           └─────────────────────────────┼─────────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   工程实践能力        │
                              │   PyTorch / JAX      │
                              │   HuggingFace / HF   │
                              │   Sim / Real Robot   │
                              └─────────────────────┘
```

---

## 📅 第一阶段：14 天速通计划（每天 8 小时）

| 日期 | 主题 | 核心产出 |
|------|------|----------|
| **Day 1** | Python/PyTorch 速通 | 能熟练使用 PyTorch 构建模型 |
| **Day 2** | 深度学习基础 | 理解 MLP/CNN/反向传播 |
| **Day 3** | Transformer 架构 | 手写一个 Mini Transformer |
| **Day 4** | NLP 与大语言模型 | 理解 LLM 原理 + LoRA 微调 |
| **Day 5** | 计算机视觉与 ViT | 理解视觉 Transformer + CLIP |
| **Day 6** | 具身智能 + 机器人运动学 | 理解 FK/IK/DH/雅可比 + MuJoCo入门 |
| **Day 7** | 行为克隆与模仿学习 | 能实现简单的 BC 算法 |
| **Day 8** | VLA 模型（RT-1, RT-2）| 理解 VLA 核心思想 |
| **Day 9** | OpenVLA 深入 | 能部署和推理 OpenVLA |
| **Day 10** | 扩散策略 & ACT | 理解动作生成的两种范式 |
| **Day 11** | 实战项目 1 + MuJoCo | OpenVLA部署+微调+仿真环境搭建 |
| **Day 12** | 实战项目 2 | ACT 训练 + 评估 |
| **Day 13** | 费曼复习日 | 完整知识体系梳理 |
| **Day 14** | 综合实战 + 规划 | 综合项目 + 自评 + 下一步 |

---

## 📚 第二阶段：30 天深入学习（速通后展开）

| 周次 | 主题 | 内容 |
|------|------|------|
| 第 3 周 | 3D 视觉与点云 | PointNet, 3D Conv, NeRF |
| 第 4 周 | 强化学习 | PPO, SAC, RL for Robotics |
| 第 5 周 | 扩散模型原理 | DDPM, DDIM, Score-based |
| 第 6 周 | 多模态架构深入 | Cross-attention, 多模态融合 |
| 第 7 周 | 先进 VLA 模型 | Octo, π₀, GR00T 论文精读 |
| 第 8 周 | 开源项目贡献 | OpenVLA 社区贡献 |

---

## ✅ 打卡规则

- [ ] 每天完成学习后在对应日期的 checkbox 打勾
- [ ] 每天完成后写 3-5 句"费曼输出"：用自己的话解释今天学到的核心概念
- [ ] 每个模块结束做一次小测验自检
- [ ] 遇到卡点记录下来，第二阶段攻克

---

## 📝 费曼学习法四步

1. **学习**: 阅读资料、视频、代码，理解核心概念
2. **讲述**: 假装给一个完全不懂的人讲解，用最简单的语言
3. **发现缺口**: 讲不清楚的地方，就是需要回头再学的地方
4. **简化和类比**: 找到概念之间的关联，用类比加深理解

> 每个文档末尾都有"费曼挑战"环节，请务必认真完成！

---

## 🔧 环境准备（Day 0 提前做）

```bash
# 1. 安装 Miniconda
# 下载: https://docs.conda.io/en/latest/miniconda.html

# 2. 创建环境
conda create -n vla python=3.10 -y
conda activate vla

# 3. 核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft
pip install numpy matplotlib jupyter tqdm
pip install gymnasium mujoco robosuite  # 仿真环境

# 4. 克隆关键项目
git clone https://github.com/openvla/openvla
git clone https://github.com/tonyzhaozh/act
git clone https://github.com/google-research/robotics_transformer
```

---

**开始学习吧！从 [[01-速通阶段/Day01-Python与PyTorch基础|Day 1 — Python 与 PyTorch 基础]] 出发！** 🚀

---

## 🔗 与知识库联动

本学习路径与 Evo-Matrix 知识库的其他模块深度绑定，在学习过程中随时查阅：

### 理论学习对应

| 学习路径模块 | 知识库中对口资源 |
|-------------|-----------------|
| Day 2 深度学习基础 | [[../../01-基础理论/深度学习基础/深度学习基础\|深度学习基础]] / [[../../01-基础理论/深度学习基础/PyTorch实战\|PyTorch实战]] |
| Day 3 Transformer | [[../../01-基础理论/Transformer与注意力/Transformer与注意力\|Transformer与注意力]] / [[../../01-基础理论/Transformer与注意力/图解Transformer\|图解Transformer]] |
| Day 4 LLM & 微调 | [[../../01-基础理论/大语言模型与微调/大语言模型与微调\|大语言模型与微调]] / [[../../01-基础理论/大语言模型与微调/LoRA与参数高效微调\|LoRA与PEFT]] |
| Day 5 ViT & 视觉 | [[../../01-基础理论/视觉基础模型/ViT/ViT概述\|ViT概述]] / [[../../01-基础理论/视觉基础模型/CLIP/CLIP与多模态对齐\|CLIP与多模态对齐]] |
| Day 6 机器人运动学 | [[../../01-基础理论/机器人运动学\|机器人运动学]] / [[../../01-基础理论/3D视觉/3D视觉与点云\|3D视觉与点云]] |
| Day 7 模仿学习 | [[../模仿学习\|模仿学习]] / [[../行为克隆\|行为克隆]] |
| Day 8-9 VLA 模型 | [[../VLA模型总览\|VLA模型总览]] / [[../VLA方向综述\|VLA方向综述]] |
| Day 10 扩散策略 | [[../扩散策略\|扩散策略]] / [[../../01-基础理论/扩散模型基础/DDPM与Score-Based\|DDPM基础]] |

### 论文精读联动

| 学习节点 | 推荐精读论文 |
|---------|------------|
| Day 8 (RT-1/RT-2) → | [[../../02-VLA/论文精读/pi系列/|π 系列论文]] |
| Day 10 (扩散策略) → | [[../../01-基础理论/扩散模型基础/条件扩散与Classifier-Free\|条件扩散]] |
| 第二阶段 (先进VLA) → | [[../../02-VLA/论文精读/|VLA 论文精读目录]]（22+ 篇已分析论文） |
| 框架优化 → | [[../../02-VLA/论文精读/框架与优化/|框架与优化模块]]（χ₀ 等后训练优化方法） |

### 项目代码模板

项目实战的代码框架可参考：
- `Mini-VLA` 全栈代码见 [[01-速通阶段/Day14-综合实战与下一步规划|Day 14]]
- VLA 成员笔记见 [[../成员笔记/|成员笔记目录]]
- 论文复现见 [[../模型复现/|模型复现目录]]

---

## 🔧 环境准备（Day 0 提前做）

```bash
# 1. 安装 Miniconda
# 下载: https://docs.conda.io/en/latest/miniconda.html

# 2. 创建环境
conda create -n vla python=3.10 -y
conda activate vla

# 3. 核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft
pip install numpy matplotlib jupyter tqdm
pip install gymnasium mujoco robosuite  # 仿真环境

# 4. 克隆关键项目
git clone https://github.com/openvla/openvla
git clone https://github.com/tonyzhaozh/act
git clone https://github.com/google-research/robotics_transformer
```
