# 基础理论

> VLA 研究的理论地基。按主题组织，每个主题一个文件夹，包含核心概念、代码实战、学习资源和自检问题。

## 学习路线

```
第 1 周：深度学习基础 → 理解神经网络如何从数据中学习
第 2 周：Transformer → 当代 AI 的核心引擎
第 3 周：视觉（ViT + 3D视觉）→ 机器人如何感知世界
第 4 周：LLM + 微调 → 语言理解和任务推理
第 5-6 周：RL + 运动学 + 扩散模型 → 动作生成与控制
```

## 目录结构

```
01-基础理论/
├── README.md                    ← 你在这里
│
├── 深度学习基础/                 ← 神经网络、反向传播、训练技巧
│   ├── 深度学习基础.md           (核心概念)
│   ├── 训练技巧与正则化.md       (Normalization/激活/调度/防过拟合)
│   └── PyTorch实战.md           (完整训练循环代码)
│
├── Transformer与注意力/          ← Self-Attention 到 VLA 应用
│   ├── Transformer与注意力.md    (核心公式 + Multi-Head + 完整实现)
│   ├── 位置编码与归一化.md       (Sinusoidal/Learnable/RoPE + Pre-norm/Post-norm)
│   └── Cross-Attention与VLA应用.md (模态融合 + RT-1/RT-2/OpenVLA 架构)
│
├── 计算机视觉与ViT/              ← CNN 到 ViT 到视觉编码器
│   ├── 计算机视觉与ViT.md       (范式转变 + ViT架构 + VLA中的视觉)
│   ├── CLIP与多模态对齐.md      (对比学习 + SigLIP vs CLIP + OpenVLA配置)
│   └── DINOv2与自监督视觉.md    (自监督 + 3D几何理解潜力)
│
├── 3D视觉/                       ← 深度/点云/NeRF/3DGS
│   ├── 3D视觉与点云.md          (传感器 + 点云/PointNet + 前沿Pipeline)
│   ├── NeRF神经辐射场.md        (体渲染 + 位置编码 + 机器人应用)
│   └── 3DGS.md                  (显式表示 + 实时渲染 + affordance/VLA应用)
│
├── 大语言模型与微调/             ← LLM 到 VLA Backbone
│   ├── 大语言模型与微调.md      (自回归 + Tokenizer + Scaling Law + VLA范式)
│   ├── LoRA与参数高效微调.md    (LoRA/QLoRA/Adapter/DoRA + 选型)
│   └── QLoRA完整实现.md         (4-bit量化 + PEFT + 训练脚本)
│
├── 强化学习.md                   ← MDP/PPO/SAC/Sim-to-Real/Reward Design
├── 机器人运动学.md               ← FK/IK/DH参数/雅可比/奇异性
│
└── 扩散模型基础/                 ← VLA 动作生成的基础
    ├── DDPM与Score-Based.md     (扩散/去噪 + 得分匹配)
    └── 条件扩散与Classifier-Free.md (文本条件 + CFG + Diffusion Policy)
```

## 主题速查

| 我想理解... | 从这里开始 |
|------------|-----------|
| 神经网络为什么能学习 | [[深度学习基础/深度学习基础|深度学习基础]] |
| VLA 的训练循环长什么样 | [[深度学习基础/PyTorch实战|PyTorch实战]] |
| Attention 的 Q/K/V 是什么 | [[Transformer与注意力/Transformer与注意力|Transformer与注意力]] |
| 视觉特征怎么进入 LLM | [[Transformer与注意力/Cross-Attention与VLA应用|Cross-Attention与VLA应用]] |
| ViT 怎么"看"图像的 | [[计算机视觉与ViT/计算机视觉与ViT|计算机视觉与ViT]] |
| SigLIP vs CLIP 怎么选 | [[计算机视觉与ViT/CLIP与多模态对齐|CLIP与多模态对齐]] |
| 点云怎么用神经网络处理 | [[3D视觉/3D视觉与点云|3D视觉与点云]] |
| NeRF 的原理 | [[3D视觉/NeRF神经辐射场|NeRF神经辐射场]] |
| 3DGS 为什么比 NeRF 快 | [[3D视觉/3DGS|3DGS]] |
| LoRA 怎么省显存 | [[大语言模型与微调/LoRA与参数高效微调|LoRA与参数高效微调]] |
| QLoRA 代码怎么写 | [[大语言模型与微调/QLoRA完整实现|QLoRA完整实现]] |
| PPO 怎么工作的 | [[../01-基础理论/强化学习|强化学习]] |
| 机械臂怎么从数字到运动 | [[../01-基础理论/机器人运动学|机器人运动学]] |
| 扩散模型怎么生成动作 | [[扩散模型基础/DDPM与Score-Based|DDPM与Score-Based]] |

## 关联方向

- [[../02-VLA/VLA方向综述|VLA 方向综述]] — 基础理论的上层应用
- [[../03-Affordance/Affordance方向综述|Affordance 综述]] — 3D 视觉 + ViT 的直接下游
- [[../05-生成式模型/生成式模型方向综述|生成式模型综述]] — 扩散模型 → 动作生成
