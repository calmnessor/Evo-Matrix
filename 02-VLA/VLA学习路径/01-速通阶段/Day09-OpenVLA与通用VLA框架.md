# 🌍 Day 9: OpenVLA 与通用 VLA 框架（8 小时）

> **口号**: "亲手跑起来一个真正的 VLA 模型——从代码到推理！"  
> **目标**: 理解 OpenVLA 架构，能下载部署并运行推理  
> **为什么重要**: OpenVLA 是目前最流行的开源 VLA，是实习/研究的最佳起点

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | OpenVLA 架构全景 | 1.5h | [ ] |
| 2 | 环境搭建与模型加载 | 1.5h | [ ] |
| 3 | 推理流程逐行拆解 | 1.5h | [ ] |
| 4 | 代码走读 | 1.5h | [ ] |
| 5 | 其他开源 VLA 概览 | 1h | [ ] |
| 6 | 费曼挑战 | 1h | [ ] |

---

## 模块 1: OpenVLA 架构全景（1.5 小时）

### 1.1 OpenVLA 是什么？ — 20min

```
OpenVLA (Open Vision-Language-Action Model):
  机构: Stanford, UC Berkeley, CMU, TU Darmstadt 等
  发布: 2024 年 6 月
  论文: "OpenVLA: An Open-Source Vision-Language-Action Model"

定位: "开源版的 RT-2"
  - 7B 参数（Vs RT-2 的 55B，小得多但也可用）
  - 完全开源（权重、代码、数据都公开）
  - 在 Open X-Embodiment 上训练

核心组成:
  视觉 Backbone: SigLIP (400M params) + DINOv2 (300M params)
  语言 Backbone: Llama-2 (7B params)
  投影层: MLP 将视觉特征映射到 LLM 空间
  
总参数量: ~7.6B
```

### 1.2 为什么是 OpenVLA？ — 15min

```
对比其他 VLA:

RT-2: 
  ✅ 效果最好
  ❌ 代码闭源，模型未公开
  
Octo:
  ✅ 代码开源
  ❌ 基于扩散模型（架构不同）
  ❌ 参数较小 (~100M)

OpenVLA:
  ✅ 完全开源（代码+权重+数据）
  ✅ 使用标准 LLM backbone（Llama-2）
  ✅ 社区活跃（GitHub 4K+ stars）
  ✅ 容易微调和扩展
  ✅ 论文详细记录了所有训练细节
```

### 1.3 完整架构图 — 30min

```
OpenVLA 的完整数据流:

┌─────────────────────────────────────────────────────────┐
│                      输入                                 │
│  ┌──────────────┐              ┌──────────────────────┐  │
│  │  摄像头图像    │              │  自然语言指令          │  │
│  │  [H, W, 3]   │              │  "pick up the ..."   │  │
│  └──────┬───────┘              └──────────┬───────────┘  │
│         │                                 │              │
│  ┌──────▼───────┐              ┌──────────▼───────────┐  │
│  │  SigLIP ViT  │              │  Llama-2 Tokenizer   │  │
│  │  (400M)      │              │                      │  │
│  └──────┬───────┘              └──────────┬───────────┘  │
│         │ [N, 1152]                       │ text_ids     │
│         │                                 │              │
│  ┌──────▼───────┐                          │              │
│  │  DINOv2 ViT  │                          │              │
│  │  (300M)      │                          │              │
│  └──────┬───────┘                          │              │
│         │ [N, 768]                         │              │
│         │                                  │              │
│  ┌──────▼───────┐                          │              │
│  │  Concat      │                          │              │
│  │  [N, 1920]   │                          │              │
│  └──────┬───────┘                          │              │
│         │                                  │              │
│  ┌──────▼───────┐                          │              │
│  │  MLP 投影    │← 把 1920 维映射到       │              │
│  │  [N, 4096]   │  Llama-2 的 4096 维     │              │
│  └──────┬───────┘                          │              │
│         │ vision_tokens [N, 4096]          │              │
│         │                       text_emb [M, 4096]      │
│         │                                  │              │
│         └──────────┬───────────────────────┘              │
│                    │                                      │
│         ┌──────────▼──────────┐                          │
│         │   Llama-2 (7B)      │                          │
│         │   32 Transformer    │                          │
│         │   Blocks            │                          │
│         │                     │                          │
│         │  视觉 token + 文本   │                          │
│         │  token 拼接输入      │                          │
│         └──────────┬──────────┘                          │
│                    │                                      │
│         ┌──────────▼──────────┐                          │
│         │   Action Head       │                          │
│         │   从特殊 token 位置  │                          │
│         │   输出 7-DOF 动作    │                          │
│         └─────────────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

### 1.4 关键架构细节 — 20min

```python
# OpenVLA 配置的伪代码
config = {
    "vision_backbone": {
        "siglip": "google/siglip-so400m-patch14-384",
        "dinov2": "facebook/dinov2-giant",
    },
    "llm_backbone": "meta-llama/Llama-2-7b-hf",
    "vision_feature_dim": 1920,  # 1152 + 768
    "llm_hidden_dim": 4096,
    "projector": "MLP(1920 → 4096)",
    "normalization": "LayerNorm before LLM input",
}
```

---

## 模块 2: 环境搭建与模型加载（1.5 小时）

### 2.1 环境配置 — 30min

```bash
# 创建环境
conda create -n openvla python=3.10 -y
conda activate openvla

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 克隆 OpenVLA
git clone https://github.com/openvla/openvla.git
cd openvla

# 安装依赖
pip install -e .

# 核心依赖
pip install transformers>=4.40.0
pip install accelerate
pip install peft
pip install tensorflow  # OXE 数据处理需要
pip install tensorflow-datasets
pip install rlds  # RLDS 数据格式

# 验证安装
python -c "import torch; print(torch.cuda.is_available())"  # 应该输出 True
python -c "from prismatic import load; print('OpenVLA imported!')"
```

### 2.2 下载模型 — 30min

```python
# 方法 1: HuggingFace 下载（推荐）
from huggingface_hub import snapshot_download

# OpenVLA 模型 (~15GB)
model_path = snapshot_download(
    "openvla/openvla-7b",
    local_dir="./openvla-7b",
    ignore_patterns=["*.msgpack", "*.h5"],  # 跳过不需要的大文件
)

# 方法 2: 直接用 transformers 加载
from transformers import AutoModelForVision2Seq, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,  # 半精度 ~14GB 显存
    device_map="auto",
    low_cpu_mem_usage=True,
)

# 方法 3: 4-bit 加载（如果显存不够）
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=quantization_config,
    device_map="auto",
)
```

### 2.3 显存规划 — 15min

```
OpenVLA-7B 显存需求:

全精度 (fp32):  ~28GB  ← 需要 A100/A6000
半精度 (bf16):  ~14GB  ← RTX 3090/4090 可以
8-bit 量化:     ~8GB   ← RTX 3070/3080 可以
4-bit 量化:     ~5GB   ← 大部分消费级 GPU 可以

如果显存不够:
  1. 用 4-bit 量化（损失 <1% 性能）
  2. 减小输入图像分辨率
  3. 使用 CPU offload
```

---

## 模块 3: 推理流程逐行拆解（1.5 小时）

### 3.1 完整推理脚本 — 40min

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# =========== 1. 加载模型 ===========
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,  # OpenVLA 有自定义代码
)
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
)

# =========== 2. 准备输入 ===========
# 加载一张机器人视角的图像
image = Image.open("robot_view.jpg").convert("RGB")

# 语言指令
instruction = "pick up the red block on the table"

# Processor 处理（自动做 resize、归一化、tokenize）
inputs = processor(
    images=image,
    text=instruction,
    return_tensors="pt",
).to(model.device)

# inputs 包含:
# - pixel_values: [1, 3, 384, 384] 或 [1, 3, 224, 224]
# - input_ids: [1, seq_len]  文本 token
# - attention_mask: [1, seq_len]

print(f"Image tensor shape: {inputs['pixel_values'].shape}")
print(f"Text tokens shape: {inputs['input_ids'].shape}")

# =========== 3. 推理 ===========
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,       # 生成的最大 token 数
        do_sample=False,          # 贪心解码（确定性）
        temperature=1.0,
    )

# =========== 4. 解码输出 ===========
# 输出包含文本 token 和动作 token
output_text = processor.decode(outputs[0], skip_special_tokens=False)
print(f"Raw output: {output_text}")

# 解析动作值
# OpenVLA 的输出格式通常是:
# <ACT> dim0_bin dim1_bin ... dim6_bin </ACT>
# 需要自定义解析函数

def parse_action_from_output(output_text):
    """从模型输出文本中提取动作值"""
    import re

    # 查找 <ACT> ... </ACT> 之间的 token
    act_match = re.search(r"<ACT>(.*?)</ACT>", output_text)
    if act_match:
        token_ids_str = act_match.group(1).strip().split()
        # 把 token_id 转回连续值
        # (具体实现取决于 OpenVLA 的 action tokenization 方案)
        # 这里展示概念
        action = []
        for tid_str in token_ids_str:
            bin_val = int(tid_str)
            # 反量化: bin → [-1, 1]
            continuous = (bin_val / 128.0) - 1.0
            action.append(continuous)
        return torch.tensor(action)
    return None

action = parse_action_from_output(output_text)
print(f"Parsed action: {action}")  # e.g., [dx, dy, dz, dr, dp, dy, grip]
```

### 3.2 动作输出详解 — 30min

```python
# OpenVLA 的输出格式（基于训练的 action space）

# 输出是 7 维向量:
# [dx, dy, dz, droll, dpitch, dyaw, gripper]
#
# 对于大多数 OXE 数据集，这是增量末端执行器控制:
#   dx, dy, dz: 末端执行器的平移量（增量）
#   droll, dpitch, dyaw: 末端执行器的旋转量（增量）
#   gripper: 夹爪开合 (0=close, 1=open)

def decode_openvla_action(generated_tokens, action_dim=7, num_bins=256):
    """
    将 OpenVLA 生成的 token 转为动作向量
    
    注意: 不同的 checkpoint 可能有不同的 action space
    需要查看训练配置！
    """
    # 找到 action token 的位置
    action_start_token = processor.tokenizer.encode("<ACT>")[1]  # 忽略 BOS
    action_end_token = processor.tokenizer.encode("</ACT>")[1]

    action_values = []

    for i in range(action_dim):
        # 每个维度的 token ID 的范围是:
        # [VOCAB_BASE + i*num_bins, VOCAB_BASE + (i+1)*num_bins)
        bin_token = generated_tokens[action_start_pos + 1 + i]
        bin_index = bin_token - (VOCAB_BASE + i * num_bins)

        # 反归一化到 [-1, 1]
        value = (bin_index / (num_bins - 1)) * 2 - 1

        # 应用数据集的归一化（如果训练时做了）
        # value = value * action_std + action_mean

        action_values.append(value)

    return torch.tensor(action_values)
```

### 3.3 常见问题与调试 — 20min

```python
# 问题 1: 输出乱码
# → 检查输入图像是否正确resize到模型期望的尺寸
print(f"Expected image size: {processor.image_processor.size}")

# 问题 2: 输出动作全是 0
# → 可能模型加载了权重但没正确初始化
# → 检查是否 load_in_4bit 时出问题

# 问题 3: 显存不足
# → 减小 batch_size 或图像分辨率
# → 使用 4-bit 量化
# → 打开 gradient checkpointing
model.gradient_checkpointing_enable()

# 问题 4: 模型输出的动作不合理
# → OpenVLA 在 OXE 上训练，不同数据集有不同的 action space
# → 需要查看模型的 dataset_statistics.json
import json
with open("openvla-7b/dataset_statistics.json") as f:
    stats = json.load(f)
    print(stats["action_mean"], stats["action_std"])
```

---

## 模块 4: 代码走读（1.5 小时）

### 4.1 项目结构 — 20min

```
openvla/
├── prismatic/                  ← 核心库
│   ├── models/
│   │   ├── vlms.py            ← VLM 模型定义
│   │   └── action_head.py     ← 动作输出头
│   ├── vla/
│   │   ├── vision_backbone/   ← 视觉编码器
│   │   │   ├── siglip.py
│   │   │   └── dinov2.py
│   │   └── action_tokenizer.py ← 动作 tokenization
│   ├── training/
│   │   ├── train.py           ← 训练入口
│   │   └── data.py            ← 数据加载
│   └── util/
├── scripts/                   ← 训练/推理脚本
├── vla-scripts/
│   ├── deploy.py              ← 部署脚本
│   └── finetune.py            ← 微调脚本
└── experiments/               ← 实验配置
```

### 4.2 核心代码走读 — 40min

```python
# ===== 视觉特征提取 =====
# 位于 prismatic/vla/vision_backbone/

class OpenVLAVisionBackbone(nn.Module):
    """
    双编码器视觉 backbone: SigLIP + DINOv2
    这是 OpenVLA 相对于其他 VLA 的关键不同
    """
    def __init__(self):
        self.siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-giant")

        # 冻结视觉编码器（训练时通常冻结）
        for p in self.siglip.parameters():
            p.requires_grad = False
        for p in self.dinov2.parameters():
            p.requires_grad = False

    def forward(self, pixel_values):
        # SigLIP 特征 [B, N_patches, 1152]
        siglip_feat = self.siglip(pixel_values).last_hidden_state

        # DINOv2 特征 [B, N_patches, 768]
        dinov2_feat = self.dinov2(pixel_values).last_hidden_state

        # 拼接 → [B, N_patches, 1920]
        fused = torch.cat([siglip_feat, dinov2_feat], dim=-1)

        return fused


# ===== 投影层 =====
class VisionProjector(nn.Module):
    """将视觉特征投影到 LLM 的嵌入空间"""
    def __init__(self, vision_dim=1920, llm_dim=4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vision_features):
        return self.projector(vision_features)


# ===== 动作头 =====
class ActionHead(nn.Module):
    """从 LLM 输出 hidden state 预测动作"""
    def __init__(self, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, hidden_states):
        # 取最后一个 token 的输出
        # (或者特定位置，取决于设计)
        last_hidden = hidden_states[:, -1, :]  # [B, 4096]
        return self.head(last_hidden)           # [B, 7]
```

### 4.3 理解 forward 的完整流程 — 25min

```python
# OpenVLA 的完整 forward（简化版，理解用）

class OpenVLAForActionPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = OpenVLAVisionBackbone()
        self.vision_projector = VisionProjector()
        self.llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.action_head = ActionHead()

        # 冻结视觉 backbone
        for p in self.vision_backbone.parameters():
            p.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        pixel_values: [B, C, H, W]  图像
        input_ids: [B, T]           文本 token（包括指令）
        attention_mask: [B, T]
        """
        B = pixel_values.shape[0]

        # 1. 视觉编码
        vision_feat = self.vision_backbone(pixel_values)  # [B, N, 1920]

        # 2. 投影到 LLM 空间
        vision_emb = self.vision_projector(vision_feat)   # [B, N, 4096]

        # 3. 获取文本嵌入
        text_emb = self.llm.model.embed_tokens(input_ids) # [B, T, 4096]

        # 4. 拼接视觉和文本 token
        # 格式: [vision_tokens | text_tokens]
        combined_emb = torch.cat([vision_emb, text_emb], dim=1)  # [B, N+T, 4096]
        combined_mask = torch.cat([
            torch.ones(B, vision_emb.shape[1], device=input_ids.device),
            attention_mask,
        ], dim=1)

        # 5. 通过 LLM
        # 注意: 这里只取最后一层输出，不生成
        outputs = self.llm.model(
            inputs_embeds=combined_emb,
            attention_mask=combined_mask,
        )
        hidden_states = outputs.last_hidden_state  # [B, N+T, 4096]

        # 6. 预测动作
        action = self.action_head(hidden_states)    # [B, 7]

        return action
```

---

## 模块 5: 其他开源 VLA 概览（1 小时）

### 5.1 Octo — 20min

```
Octo (Octopus):
  机构: UC Berkeley, 2024
  定位: 开源通用机器人策略

与 OpenVLA 的不同:
  - 基于扩散模型（不是自回归 LLM）
  - 较小（~100M vs 7B）
  - 推理快得多
  - 代码: https://github.com/octo-models/octo

架构:
  图像 → 扩散 Transformer (DiT) → 动作序列
  用文本做条件注入（类似 Stable Diffusion）
```

### 5.2 π₀ (Pi-Zero) — 15min

```
π₀:
  机构: Physical Intelligence, 2024
  定位: 大规模通用 VLA

亮点:
  - 训练规模巨大（不公开具体数字）
  - 展示了在多种机器人上的泛化
  - 可以完成复杂操作（叠衣服、整理桌面）

架构思路:
  预训练 VLM (PaliGemma) + 流匹配 (Flow Matching)
  流匹配 = 扩散模型的连续时间推广
```

### 5.3 GR00T (NVIDIA) — 10min

```
GR00T (Generalist Robot 00 Technology):
  机构: NVIDIA, 2024
  定位: 面向工业人形机器人的基础模型

特点:
  - 多模态输入（视觉、语言、本体感知）
  - 输出人形机器人全身动作
  - 在 Isaac Sim 中大规模仿真训练
  - 工业级部署（延迟、鲁棒性要求高）
```

### 5.4 快速对比 — 15min

| 模型 | 参数 | 架构 | 开源 | 特点 |
|------|------|------|------|------|
| RT-1 | ~50M | Transformer+CNN | ❌ | 先驱 |
| RT-2 | 55B | VLM+Action Token | ❌ | 效果最好 |
| OpenVLA | 7B | SigLIP+DINOv2+Llama | ✅ | 最流行开源 |
| Octo | ~100M | Diffusion Transformer | ✅ | 轻量快速 |
| π₀ | 未公开 | VLM+Flow Matching | ❌ | 最强泛化 |
| ACT | ~100M | CVAE+Transformer | ✅ | 精细操作 |

---

## 🎤 费曼挑战（1 小时）

### 任务 1: 代码讲解 (25min)
关掉所有代码，口头讲解 OpenVLA 的 forward 函数:
1. 视觉特征怎么提取？（两个编码器，怎么融合）
2. 视觉特征怎么和文本 token 拼在一起？
3. 动作怎么从 hidden state 得到？

**录下来，听一遍，看哪里卡住了。**

### 任务 2: 架构对比 (20min)
画一张表，对比 RT-1, RT-2, OpenVLA, Octo, ACT 的:
- 架构类型
- 动作表示方式
- 模型大小
- 开源情况

### 任务 3: 环境验收 (15min)
确认你能成功:
- [ ] 导入 OpenVLA
- [ ] 加载模型（即使是 4-bit）
- [ ] 运行一次 inference
- [ ] 理解输出的含义

---

## 📝 今日自检清单

- [ ] 我能画出 OpenVLA 的完整架构
- [ ] 我理解为什么用 SigLIP + DINOv2 双编码器
- [ ] 我理解视觉投影层的作用
- [ ] 我知道视觉 token 和文本 token 怎么拼接
- [ ] 我成功安装并加载了 OpenVLA
- [ ] 我至少跑通了一次推理
- [ ] 我读过 OpenVLA 的核心代码（vision_backbone, action_head）
- [ ] 我了解 Octo、π₀、GR00T 的基本特点

---

> ✅ **完成打卡**: 开源 VLA 已上手！明天学习动作生成的另一种范式——扩散策略！
>
> 🔜 **明日预告**: Day 10 — 扩散策略与 ACT 深入，理解去噪过程如何生成动作！
