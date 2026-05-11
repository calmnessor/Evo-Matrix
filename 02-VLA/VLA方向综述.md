# VLA 方向综述：从 RT-1 到 π*₀.₆

> Vision-Language-Action — 构建"看→理解→行动"的端到端机器人基础模型。本文为科研新手提供该方向的完整学术地图。

## 1. 定义与直觉

**一句话**：VLA 是一个神经网络，输入摄像头画面和自然语言指令，直接输出机器人关节/末端动作。

**生活化类比**：传统机器人是"翻译官"——感知模块（眼睛）→ 规划模块（大脑）→ 控制模块（手），三班人马各说各的语言，信息传递有损失。VLA 是"母语者"——看、想、做在同一个神经网络里完成，从像素直达动作。

```
传统: Camera → 物体检测 → 场景理解 → 任务规划 → 运动规划 → 控制 → Motor
      [独立模块]   [独立模块]   [独立模块]   [独立模块]   [独立模块]

VLA:  Camera + "把杯子拿过来" → [ 一个大模型 ] → Motor
```

## 2. 为什么重要

具身智能的终极目标是让机器人像人一样通用——走进陌生环境，听懂自然语言指令，完成各种操作任务。VLA 是实现这个目标最直接的路径：

| 传统方法的根本缺陷 | VLA 的解法 |
|------------------|-----------|
| 模块间信息损失（检测错误→规划失败） | 端到端学习，信息全保留 |
| 每个任务需单独编程 | 一个 prompt 换一个任务 |
| 面对未见物体/场景完全失效 | LLM 常识推理 → 泛化 |
| 数据需求专业化（需要专家标注） | 利用互联网级预训练知识 |

**学术定位**：VLA 是连接 CV（视觉）、NLP（语言）、Robotics（控制）三个领域的交叉点。它从 CV 借视觉编码器，从 NLP 借 LLM，从 Robotics 借动作生成和物理约束。在这三者的交界处，VLA 创造了一个全新的研究范式。

## 3. 技术演进时间线

```
2016-2020: 前 VLA 时代
├─ 2016: BC + 监督学习 → 简单任务
├─ 2018: QT-Opt (RL + 大规模数据)
├─ 2019: Causal InfoGAN (端到端抓取)
└─ 2020: ———

2021-2022: VLA 萌芽期
├─ 2021: CLIPPort (CLIP + 机器人操作，CV↔Robot 首次打通)
├─ 2022: SayCan (LLM 规划 + RL 执行, Google)
├─ 2022: Gato (多模态通才 agent, DeepMind)
├─ 2022.12: RT-1 (首次大规模 Transformer 机器人控制, Google)
│          → 130K 真实场景轨迹, TokenLearner 压缩视觉
└─ 2022: ——

2023: VLA 概念诞生与爆发
├─ 2023.03: PaLM-E (LLM 直接输出机器人动作, Google)
├─ 2023.07: RT-2 (VLA 术语正式诞生! Google)
│          → "把动作也当 token" → VLA = VLM + Action Tokenization
├─ 2023.04: ACT (CVAE + Action Chunking, Stanford)
├─ 2023.03: Diffusion Policy (扩散生成动作, Columbia)
└─ 2023.10: Open X-Embodiment (1M+ 轨迹开源, 34 个实验室)

2024: 工业级 VLA
├─ 2024.06: Octo (开源通用机器人模型, Berkeley)
├─ 2024.06: OpenVLA (开源 7B VLA, Stanford)
├─ 2024.10: π₀ (Flow Matching + MoE + 10K hrs, Physical Intelligence)
├─ 2024.11: GR00T (NVIDIA 工业级基础模型)
└─ 2024: RDT (扩散 Transformer + 双臂, 清华)

2025: VLA 深度优化与 RL 进化
├─ 2025.01: π₀-FAST (DCT+BPE 动作压缩 10×)
├─ 2025.02: Hi Robot (System-1/2 分层推理)
├─ 2025.03: Gemini Robotics (Google, Gemini 2.0 + 具身推理)
├─ 2025.04: π₀.₅ (开放世界泛化, 100 家庭)
├─ 2025.05: π₀.₅-KI (知识绝缘, 训练快 7.5×)
├─ 2025.06: RTC (实时动作分块, NeurIPS 2025)
├─ 2025.11: π*₀.₆ / RECAP (RL 自进化, 13h 连续运行)
└─ 2025.12: π₀.₅+ego (人→机器人技能迁移涌现)
```

## 4. 方法分类学

```
VLA 模型
│
├── 路线 A: 离散动作 Token 化 (Action Tokenization)
│   思路: 把连续动作离散化成 token → 自回归生成
│   优势: 与 LLM 训练/推理完全兼容, 多任务自然泛化
│   劣势: 量化精度损失, 自回归推理慢
│   代表: RT-2, OpenVLA, RoboFlamingo, Octo
│
├── 路线 B: 扩散/流匹配 动作生成 (Diffusion/Flow)
│   思路: 从噪声逐步去噪 → 生成连续动作序列
│   优势: 表达多模态分布, 动作平滑
│   劣势: 多步去噪推理慢, 与 LLM pipeline 不完全兼容
│   代表: Diffusion Policy, RDT, π₀ (Flow Matching)
│
├── 路线 C: CVAE 动作生成
│   思路: Encoder→Latent→Decoder, 从 latent 一次生成多步动作
│   优势: 一次推理出多步 (chunking), 动作稳定
│   劣势: VAE prior 可能欠拟合, 多样性有限
│   代表: ACT, ACT++
│
├── 路线 D: 分层推理架构
│   思路: System-2 (VLM 慢推理) → System-1 (快速执行)
│   优势: 复杂任务拆解, 用户可实时纠正
│   劣势: 两个模型 = 计算开销大
│   代表: Hi Robot, π₀.₅ ("Think then Act")
│
└── 路线 E: RL 自进化
    思路: BC 初始化 → RL 从真实世界经验中改进
    优势: 可超越人类专家, 持续改进
    劣势: 需要可靠 reward 信号, 安全风险
    代表: π*₀.₆ (RECAP)
```

### 路线选择指南

| 场景 | 推荐路线 |
|------|---------|
| 快速原型验证 | C (ACT) — 代码简单、训练快 |
| 多任务语言条件 | A (OpenVLA) — 开源、LoRA 微调方便 |
| 精细灵巧操作 | B (π₀/π₀.₅) — Flow Matching 精度最高 |
| 复杂长周期任务 | D (Hi Robot / π₀.₅) — 分层推理 |
| 工业级鲁棒性 | E (π*₀.₆) — RL 自进化 |

## 5. 代表工作深度对比

| 维度 | RT-2 | OpenVLA | π₀ | π₀.₅-KI | π*₀.₆ |
|------|------|--------|-----|---------|--------|
| **年份** | 2023 | 2024 | 2024 | 2025 | 2025 |
| **机构** | Google | Stanford | PI | PI | PI |
| **VLM Backbone** | PaLI-X 55B | SigLIP+LLaMA 7B | PaliGemma 3B | Gemma 2.6B | Gemma 3 4B |
| **动作生成** | 离散 token | 离散 token (256 bins) | Flow Matching | Flow Matching (KI) | Flow Matching (RECAP) |
| **训练数据** | 130K RT-1 + 网络 | 970K OXE | 10K hrs PI | 同 π₀.₅ | 同 π₀.₆ |
| **训练范式** | BC (监督) | BC (监督) | BC (Flow) | BC (KI) | BC → RL |
| **开源** | ❌ | ✅ | ❌ | 部分 | ❌ |
| **核心创新** | VLA 术语诞生 | 开源 7B | MoE + FM | stop_gradient | Advantage Conditioning |
| **推理速度** | 慢 (55B) | 慢 (7B 自回归) | 中等 (FM) | 快 (KI+FM) | 快 |
| **泛化性** | 中 | 中 | 中 | 高 (70% OOD) | 极高 (自进化) |

## 6. 当前 SOTA 与趋势 (2025)

### 已解决的问题
- ✅ 简单桌面操作（抓取+放置）在 OXE 数据上已达 ~80%+ 成功率
- ✅ 单一场景的灵巧操作已可行（π₀.₅ 在陌生家庭 ~72%）
- ✅ LoRA 微调少量数据可使 VLA 适应新任务

### 正在突破的方向
- 🔥 **RL 自进化**（π*₀.₆）：从 BC 走向 RL，超越人类专家
- 🔥 **人机迁移**（π₀.₅+ego）：人类视频 → 机器人技能，数据瓶颈的根本性突破
- 🔥 **实时推理**（RTC）：推理和执行并行，消除"发呆"时间
- 🔥 **开源基建**（OpenVLA/openpi）：VLA 从"闭源竞赛"走向"开放生态"

### 三大趋势
1. **从 BC 到 RL**：模仿学习有天花板，RL 是突破天花板的关键
2. **从单一模态到多模态融合**：视觉+语言+触觉+力觉的联合建模
3. **从实验室到开放世界**：π₀.₅ 证明了 VLA 可以在真实陌生家庭中工作

## 7. 关键挑战与开放问题

| 挑战 | 现状 | 可能的突破方向 |
|------|------|-------------|
| **数据稀缺** | 机器人数据比互联网数据少 4-5 个数量级 | 人类视频迁移, 仿真数据, 自进化 |
| **长周期可靠性** | >10 分钟任务成功率仍 <50% | 分层推理 + RL + 更好的 VLM |
| **安全性** | LLM 幻觉在物理世界有真实后果 | 形式化约束, 运行时监控, Human-in-loop |
| **推理速度** | 7B+ 模型难以 >10Hz | 模型蒸馏, 专用芯片, KI 架构 |
| **跨具身泛化** | 不同机器人需要重新适配 | 统一动作空间, 具身 token 化 |
| **触觉缺失** | 当前 VLA 几乎只用视觉 | 触觉传感器 + 多模态融合 |
| **持续学习** | 部署后策略固定不变 | RECAP 已开始探索但远未解决 |
| **可解释性** | 黑盒决策 | 注意力可视化, 语言化推理 (Hi Robot) |

## 8. 与其它方向的关系

```
                    ┌──────────────┐
                    │   世界模型     │ ← 预测动作后果, 辅助 VLA 规划
                    └──────┬───────┘
                           │ 提供想象能力
                           ↓
┌──────────┐      ┌─────────────────┐      ┌──────────┐
│ Affordance│ ──→ │      VLA        │ ←─── │  Grasp   │
│ "哪里能做" │     │ "做什么动作"     │      │ "怎么抓"  │
└──────────┘      └─────────────────┘      └──────────┘
       ↑                    ↓                     ↑
       │            ┌───────────────┐             │
       └─────────── │  基础理论      │ ───────────┘
                    │ CV + NLP + RL │
                    │ + 机器人学     │
                    └───────────────┘
```

- **VLA ← Affordance**: Affordance 提供"哪里可以执行动作"的先验，VLA 可以直接条件化在 affordance map 上
- **VLA ← Grasp**: 抓取是 VLA 最常用的动作原语之一
- **VLA → 世界模型**: 世界模型预测"做完这个动作后世界是什么样"，VLA 用世界模型做规划和想象
- **VLA → 仿真/真机**: VLA 的训练在仿真中，验证在真机上，两者通过 Sim2Real 连接

## 9. 必读论文清单

### 入门级（建立核心概念）
| 论文 | 年份 | 一句话 | 阅读时间 |
|------|------|--------|---------|
| **RT-2** | 2023 | VLA 概念起源——把动作变成 token | 2h |
| **ACT** | 2023 | 最简单的 VLA 实现——BC + CVAE | 1.5h |
| **Diffusion Policy** | 2023 | 扩散模型生成动作的优雅方案 | 2h |
| **OpenVLA** | 2024 | 开源 VLA 的完整蓝图 | 2h |

### 进阶级（深入理解架构）
| 论文 | 年份 | 一句话 | 阅读时间 |
|------|------|--------|---------|
| **π₀** | 2024 | MoE + Flow Matching 工业级方案 | 3h |
| **π₀.₅-KI** | 2025 | 知识绝缘——解决 VLA 根本矛盾 | 2h |
| **Hi Robot** | 2025 | 分层推理——System-1 + System-2 | 2h |

### 前沿级（把握最新方向）
| 论文 | 年份 | 一句话 | 阅读时间 |
|------|------|--------|---------|
| **π*₀.₆ (RECAP)** | 2025 | RL 自进化——超越模仿学习天花板 | 3h |
| **π₀.₅+ego** | 2025 | 人类→机器人技能迁移涌现 | 2h |
| **RTC** | 2025 | 实时动作分块——消除推理延迟 | 1.5h |

## 10. 新手学习路径

### 第 1 周：建立直觉
- Day 1-2: 读 RT-2 博客 + ACT 论文 (理解 VLA 的基本思想)
- Day 3-4: 跑通 ACT 代码 (LeRobot 实现), 看策略 rollout
- Day 5-7: 读 VLA模型总览, 理解三大技术路线

### 第 2 周：理解核心架构
- Day 1-3: 读 OpenVLA 论文 + 源码, 理解 VLM→Action 的完整流程
- Day 4-5: 读 Diffusion Policy 论文, 理解扩散生成 vs 自回归生成
- Day 6-7: 读 π₀ 论文, 理解 Flow Matching + MoE 架构

### 第 3 周：动手实践
- Day 1-3: 在 MuJoCo 中用 BC 训练简单 reach/grasp 策略
- Day 4-5: 用 OpenVLA + LoRA 在自己的数据上微调
- Day 6-7: 在 LIBERO benchmark 上评估

### 第 4 周：跟上前沿
- Day 1-3: 读 π₀.₅-KI + π*₀.₆, 理解 KI 和 RECAP 的设计思路
- Day 4-5: 阅读 π₀.₅+ego, 思考数据瓶颈的突破方向
- Day 6-7: 写一篇 mini-review：总结你认为 VLA 领域未来 2 年最重要的 3 个方向

## 11. 资源汇总

### 开源代码
| 代码库 | 内容 | 链接 |
|--------|------|------|
| **openpi** | π₀.₅ 官方实现 | github.com/Physical-Intelligence/openpi |
| **OpenVLA** | OpenVLA 训练/推理 | github.com/openvla/openvla |
| **LeRobot** | ACT/Diffusion Policy 标准实现 | github.com/huggingface/lerobot |
| **robomimic** | BC/IRIS 等模仿学习算法 | github.com/ARISE-Initiative/robomimic |
| **LIBERO** | VLA 标准 benchmark | github.com/Lifelong-Robot-Learning/LIBERO |

### 数据集
| 数据集 | 规模 | 获取 |
|--------|------|------|
| Open X-Embodiment | 1M+ 轨迹 | gs://gresearch/robotics |
| Bridge v2 | 60K 轨迹 | HuggingFace Datasets |
| DROID | 100K 轨迹 | droid-dataset.github.io |

### 教程与课程
| 资源 | 类型 | 链接 |
|------|------|------|
| CS 285 (Sergey Levine) | 课程 | rail.eecs.berkeley.edu/deeprlcourse |
| LeRobot Tutorials | 教程 | github.com/huggingface/lerobot |
| 具身智能前沿论坛 | 中文社区 | — |

## 关联笔记

- [[VLA模型总览]] — VLA 模型谱系的详细对比
- [[📌-VLA总览]] — VLA 方向的快速导航
- [[行为克隆]] — VLA 最基础的训练方法
- [[多模态融合架构]] — V+L 融合的技术方案
- [[../03-Affordance/Affordance方向综述|Affordance 综述]] — 动作可能性的另一视角
- [[../04-Grasp/Grasp方向综述|Grasp 综述]] — 抓取专项综述
- [[../05-世界模型/世界模型方向综述|世界模型 综述]] — 预测与规划能力
