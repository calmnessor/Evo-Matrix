# HuggingFace 工具链

> VLA 开发的核心工具生态——模型加载、微调、部署

## 核心库

### Transformers
```python
from transformers import AutoModel, AutoProcessor

# 加载视觉模型
vision_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")

# 加载语言模型
from transformers import AutoModelForCausalLM
llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

### PEFT (Parameter-Efficient Fine-Tuning)
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,            # LoRA rank
    lora_alpha=32,   # Scale factor
    target_modules=["q_proj", "v_proj"],  # 目标模块
    lora_dropout=0.05,
)

model = get_peft_model(llm, lora_config)
# 只训练 ~0.1% 的参数！
```

### Datasets
```python
from datasets import load_dataset

dataset = load_dataset("path/to/robot_dataset")
```

### Accelerate
```python
# 多 GPU 训练 / 混合精度 / DeepSpeed 集成
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

## VLA 开发常用技巧

### 加载 OpenVLA
```python
# OpenVLA 基于 HF Transformers
from transformers import AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
```

### SigLIP 视觉编码
```python
from transformers import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)  # [1, 729, 1152] features
```

## 自检问题
- [ ] 我会用 Transformers 加载视觉和语言模型
- [ ] 我能用 PEFT 做 LoRA 微调
- [ ] 我知道 Accelerate 的作用

## 关联笔记
- [[大语言模型与微调]] — LoRA 原理
- [[计算机视觉与ViT]] — 视觉模型加载
- [[VLA模型总览]]
