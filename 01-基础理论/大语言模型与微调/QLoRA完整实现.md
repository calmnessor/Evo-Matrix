# QLoRA 完整实现

> 用一张 RTX 3060 微调 LLaMA-3B。完整可运行的代码模板。

---

## 1. 环境准备

```bash
pip install transformers peft bitsandbytes accelerate datasets
```

---

## 2. 最小可运行示例

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# 1. 加载量化模型 (QLoRA)
model_name = "meta-llama/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,               # 4-bit 量化
    bnb_4bit_compute_dtype=torch.bfloat16,
    device_map="auto",               # 自动分配 GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA 默认无 pad_token

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=8,                              # rank
    lora_alpha=16,                    # 缩放
    target_modules=["q_proj", "v_proj"],  # 通常只加在 Q, V 上
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,     # 自回归任务
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 3,407,872 || all params: 3,216,795,648 || trainable%: 0.1060

# 3. 准备数据 (示例: 指令格式)
data = [
    {"text": "### Instruction: 把红色方块推到蓝色圆圈的右边\n### Response: 末端移动到(0.3, 0.1, 0.05)，夹爪闭合，向右推20cm"},
    # ... 更多数据
]
dataset = Dataset.from_list(data)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

dataset = dataset.map(tokenize)

# 4. 训练
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./lora-vla",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,   # 等效 batch_size=16
        num_train_epochs=3,
        learning_rate=2e-4,              # LoRA 学习率通常比全量高
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    ),
    train_dataset=dataset,
)
trainer.train()

# 5. 保存 LoRA 权重 (仅 ~10MB!)
model.save_pretrained("./lora-vla-final")

# 6. 推理时加载
# from peft import PeftModel
# model = PeftModel.from_pretrained(base_model, "./lora-vla-final")
```

---

## 3. VLA 微调模板

```python
# VLA 数据格式示例
vla_data = [
    {
        "image": "episode_001_frame_010.jpg",
        "instruction": "pick up the red cup on the left",
        "action": [0.023, -0.015, 0.008, 0.001, 0.002, -0.003, 1.0],  # Δx, Δy, Δz, Δr, Δp, Δy, gripper
    },
    # ...
]

# 格式化后
def format_vla_sample(sample):
    return {
        "text": f"### Instruction: {sample['instruction']}\n### Action: {sample['action']}"
    }
```

---

## 4. 关键参数速查

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `r` | 8 | 低秩分解秩，越大越强 |
| `lora_alpha` | 16 | 缩放因子，实际 lr = α/r |
| `learning_rate` | 2e-4 | LoRA 比全量微调可用更大 lr |
| `batch_size` | 4 | 配合 gradient_accumulation 用 |
| `gradient_accumulation_steps` | 4 | bs=4 × ga=4 = 等效 bs=16 |
| `max_length` | 512 | VLA 序列通常 <512 tokens |
| `warmup_ratio` | 0.03 | 小幅 warmup 即可 |

---

## 关联笔记

- [[LoRA与参数高效微调]] — LoRA/QLoRA 原理和各种方法的对比
- [[大语言模型与微调]] — LLM 基础概念
- [[../深度学习基础/PyTorch实战|PyTorch实战]] — 训练循环基础
