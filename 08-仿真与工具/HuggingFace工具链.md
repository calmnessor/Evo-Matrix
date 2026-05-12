# HuggingFace 工具链

> VLA 开发的核心工具生态——从模型加载、微调到部署的完整 pipeline。

## 核心库总览

| 库 | 用途 | VLA 场景 |
|----|------|---------|
| `transformers` | 模型加载/推理 | OpenVLA, RT-2, π₀ 的 VLM backbone |
| `peft` | 参数高效微调 | LoRA/QLoRA 微调 VLA |
| `datasets` | 数据加载/处理 | OXE, Bridge v2 等数据集 |
| `accelerate` | 分布式训练 | 多 GPU 训练 VLA |
| `trl` | RL 微调 | DPO/PPO 对齐 VLA |
| `diffusers` | 扩散/流匹配 | Diffusion Policy, Flow Matching |
| `lerobot` | 机器人学习 | ACT, Diffusion Policy 的 HF 标准实现 |
| `gradio` | Demo 搭建 | VLA 在线 demo |
| `huggingface_hub` | 模型上传/下载 | 分享/获取 VLA checkpoint |

## 1. Transformers — VLA 的 VLM Backbone

### 加载视觉编码器

```python
from transformers import AutoModel, AutoProcessor
import torch

# SigLIP — OpenVLA / π₀ 使用的视觉编码器
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")

# 处理单帧
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    features = model(**inputs).last_hidden_state  # [B, 729, 1152]

# DINOv2 — 更好的空间理解（可选替代）
dino = AutoModel.from_pretrained("facebook/dinov2-large")
```

### 加载 VLA 模型

```python
# OpenVLA (基于 Prismatic VLMs)
from transformers import AutoModelForVision2Seq, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b")

# 推理
inputs = processor(images=image, text="pick up the red cup", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
actions = model.decode_actions(outputs)  # 将 token 解码为动作
```

### 加载 LLM Backbone

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# π₀.₅ 使用 Gemma 2.6B
gemma = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 生成动作 token (自回归 VLA)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
prompt = "User: Pick up the cup. Action:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = gemma.generate(**inputs, max_new_tokens=50)
```

## 2. PEFT — VLA 的参数高效微调

### LoRA 微调 VLA

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 配置（针对 VLA 优化）
lora_config = LoraConfig(
    r=16,                     # LoRA rank (越大容量越大)
    lora_alpha=32,            # 缩放因子
    target_modules=[          # 目标层
        "q_proj", "v_proj",   # Attention 的 Q/V 投影
        "k_proj", "o_proj",   # Attention 的 K/O 投影
        "gate_proj",          # FFN 门控
        "up_proj", "down_proj"  # FFN 上下投影
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 只微调 VLM backbone，冻结动作 head
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 8.4M || all params: 3.2B || trainable%: 0.26%
```

### QLoRA — 更低显存

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    quantization_config=bnb_config,
    device_map="auto",
)
model = get_peft_model(model, lora_config)
# 现在 7B 模型可以在单张 RTX 3090 上微调!
```

### 保存与加载

```python
# 只保存 adapter (几 MB)
model.save_pretrained("./vla_lora_adapter")
tokenizer.save_pretrained("./vla_lora_adapter")

# 加载
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
model = PeftModel.from_pretrained(base_model, "./vla_lora_adapter")
```

## 3. LeRobot — HF 官方的机器人学习库

LeRobot 是 HuggingFace 推出的模仿学习标准库，实现了 ACT、Diffusion Policy 等算法。

```python
from lerobot.datasets import LeRobotDataset
from lerobot.policies import ACTPolicy

# 加载数据集
dataset = LeRobotDataset("lerobot/pusht")

# 创建 ACT 策略
policy = ACTPolicy(
    input_shapes={"observation.image": [3, 480, 640]},
    output_shapes={"action": [2]},  # x, y
    n_action_steps=10,
    chunk_size=10,
)

# 训练
for batch in dataset:
    loss = policy.forward(batch)
    loss.backward()
```

### 支持的算法

| 算法 | 类 | 论文 |
|------|-----|------|
| Behavior Cloning | — | — |
| ACT | `ACTPolicy` | [2304.13705](https://arxiv.org/abs/2304.13705) |
| Diffusion Policy | `DiffusionPolicy` | [2303.04137](https://arxiv.org/abs/2303.04137) |
| TDMPC | `TDMPCPolicy` | [2203.04955](https://arxiv.org/abs/2203.04955) |
| VQ-BeT | `VQBeTPolicy` | [2306.14381](https://arxiv.org/abs/2306.14381) |

## 4. Datasets — 处理机器人数据

### 加载 OXE 数据 (RLDS)

```python
from datasets import load_dataset
import tensorflow_datasets as tfds

# 方法 1: 通过 TFDS 加载 OXE
builder = tfds.builder_from_directory(
    "gs://gresearch/robotics/bridge_v2/1.0.0"
)
dataset = builder.as_dataset(split="train")

# 方法 2: 通过 HuggingFace Datasets (如果已转换)
dataset = load_dataset("lerobot/pusht", split="train")

# 处理 (image, language, action) 三元组
for episode in dataset:
    for step in episode["steps"]:
        image = step["observation"]["image"]
        action = step["action"]
        language = step["language_instruction"]
```

### 数据预处理流水线

```python
from datasets import Dataset, Features, Image, Value, Sequence

# 自定义特征定义
features = Features({
    "observation.image": Image(),
    "observation.state": Sequence(Value("float32"), length=7),  # 7-dof
    "action": Sequence(Value("float32"), length=7),
    "language_instruction": Value("string"),
})

# 构建 dataset
def preprocess(episode):
    images = [step["image"] for step in episode]
    states = [step["state"] for step in episode]
    actions = [step["action"] for step in episode]
    return {
        "observation.image": images,
        "observation.state": states,
        "action": actions,
        "language_instruction": episode[0]["instruction"],
    }

dataset = Dataset.from_generator(
    lambda: map(preprocess, raw_episodes),
    features=features,
)
```

## 5. Accelerate — 分布式训练

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=4,
)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

# 保存 (自动处理多 GPU / DeepSpeed)
accelerator.save_model(model, "./checkpoints")
```

## 6. TRL — VLA 的 RL 对齐

```python
from trl import DPOTrainer, DPOConfig

# DPO: 让 VLA 学会偏好 (成功 > 失败)
dpo_config = DPOConfig(
    output_dir="./vla_dpo",
    per_device_train_batch_size=2,
    learning_rate=5e-6,
    max_length=2048,
)

trainer = DPOTrainer(
    model=model,
    config=dpo_config,
    train_dataset=dpo_dataset,  # (prompt, chosen_action, rejected_action)
    tokenizer=tokenizer,
)
trainer.train()
```

## 7. 模型上传与分享

```python
from huggingface_hub import HfApi, create_repo, upload_folder

# 创建 repo
create_repo("my-org/my-vla-model", private=True)

# 上传
api = HfApi()
api.upload_folder(
    folder_path="./checkpoints/step-10000",
    repo_id="my-org/my-vla-model",
    commit_message="Best open-world VLA checkpoint",
)
```

## 完整 VLA 微调流程

```python
# 1. 加载预训练 VLA (LoRA)
from transformers import AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model

model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
model = get_peft_model(model, LoraConfig(r=16, ...))

# 2. 加载微调数据
from datasets import load_dataset
dataset = load_dataset("path/to/my_robot_data")

# 3. 训练
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./vla_finetuned",
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        bf16=True,
        gradient_checkpointing=True,
    ),
    train_dataset=dataset["train"],
)
trainer.train()

# 4. 推理
model.eval()
actions = model.generate(images=test_image, text="pick the cup")
robot.execute(actions)
```

## 自检问题

### 基础关
- [ ] 我会用 `transformers` 加载 ViT/SigLIP/Gemma 模型
- [ ] 我能用 `peft` 配置 LoRA 并微调 VLA
- [ ] 我理解 `accelerate` 的 mixed precision 和 gradient accumulation
- [ ] 我能用 `datasets` 加载和处理机器人数据

### 进阶关
- [ ] 我能在 LeRobot 上跑通 ACT/Diffusion Policy 训练
- [ ] 我理解 QLoRA 的 4-bit 量化原理
- [ ] 我会用 `trl` 做 DPO 偏好对齐
- [ ] 我能把 LoRA adapter 合并到基础模型并上传 Hub

### 实战关
- [ ] 我用 HF 工具链完整微调过一个 VLA 模型
- [ ] 我处理过 OXE 数据并转换为 HuggingFace Dataset
- [ ] 我在多 GPU 上用 Accelerate 训练过 VLA
- [ ] 我在 HuggingFace Hub 上分享过模型

## 关联笔记
- [[../../01-基础理论/大语言模型与微调]] — LoRA/QLoRA 深入原理
- [[../../02-VLA/OpenX-Embodiment数据集]] — RLDS 数据处理
- [[../../02-VLA/VLA模型总览]] — 主流 VLA 模型
- [[仿真与工具索引]] — 回到工具索引
