# 📝 Day 4: NLP 与大语言模型速通（8 小时）

> **口号**: "理解语言模型，就理解了 VLA 的一半——语言理解！"  
> **目标**: 理解 GPT 架构、Tokenization、LoRA 微调  
> **为什么重要**: VLA = VLM 基础 + 动作输出，LLM 是 VLA 的语言大脑

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | Tokenization 词元化 | 1h | [ ] |
| 2 | GPT 架构与前缀语言模型 | 1.5h | [ ] |
| 3 | LLM 训练三阶段 | 1h | [ ] |
| 4 | LoRA 与 PEFT 高效微调 | 1.5h | [ ] |
| 5 | HuggingFace Transformers 实战 | 1.5h | [ ] |
| 6 | 费曼挑战 | 1.5h | [ ] |

---

## 模块 1: Tokenization 词元化（1 小时）

### 1.1 为什么需要 Tokenization？ — 15min

```
计算机不识字 → 需要把文字变成数字

原始文本: "拿起红色的积木"
    ↓ Tokenizer
Token IDs: [256, 1234, 567, 890, 12]
    ↓ Embedding
向量: [[0.1, -0.3, ...], [...], ...]   ← 这才是模型真正的输入
```

### 1.2 BPE (Byte-Pair Encoding) — 20min

现代 LLM 大多用 BPE（GPT 系列、Llama 系列都用）：

```
BPE 原理（简化）:
1. 统计所有相邻字符对的频率
2. 合并最高频的对 → 新 token
3. 重复直到 vocab_size 达到目标

示例: "low" "lower" "newest" "widest"
  初始词汇: l, o, w, e, r, n, s, t, i, d
  → 合并 "e"+"s" → "es"
  → 合并 "es"+"t" → "est"
  → 合并 "l"+"o" → "lo"
  → 合并 "lo"+"w" → "low"
  ... 最终词汇表包含常见词根、词缀
```

### 1.3 实际使用 — 15min

```python
from transformers import AutoTokenizer

# 这个 tokenizer 在 OpenVLA 里也会用到
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 编码
text = "Pick up the red block"
tokens = tokenizer.encode(text)
print(f"Token IDs: {tokens}")
# → [1, 12345, 456, 278, 3456, 7890]

print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
# → ['<s>', '▁Pick', '▁up', '▁the', '▁red', '▁block']

# 解码
decoded = tokenizer.decode(tokens)
print(decoded)  # → "Pick up the red block"
```

### 1.4 特殊 Token — 10min

```python
# VLA 中常用的特殊 token
special_tokens = {
    "bos_token": "<s>",         # 序列开始
    "eos_token": "</s>",        # 序列结束
    "pad_token": "<pad>",       # 填充
    "action_start": "<action>", # 动作开始标记（RT-2 风格）
    "action_end": "</action>",  # 动作结束标记
}
```

**自检**: 为什么 VLA 需要 `<action>` 这样的特殊 token？

---

## 模块 2: GPT 架构与因果语言模型（1.5 小时）

### 2.1 GPT vs BERT vs Encoder-Decoder — 30min

```
GPT (Decoder-Only):        当今主流 ← VLA 用这个
  输入 → [Transformer Block × N] → 输出
  特点: 单向（只看左边）、自回归生成

BERT (Encoder-Only):
  输入 → [Transformer Block × N] → 输出
  特点: 双向、适合理解任务

T5 (Encoder-Decoder):  
  输入 → [Encoder] → [Decoder] → 输出
  特点: 完整架构、适合翻译/摘要

VLA 为什么用 Decoder-Only?
  → 机器人需要"生成"动作序列，和文本生成一样是自回归任务
  → 架构简单，容易扩展到多模态
```

### 2.2 Causal Mask 因果掩码 — 30min

```python
# GPT 的关键差异: Causal Attention
# 第 i 个 token 只能看到第 0~i 个 token (不能偷看未来)

def create_causal_mask(seq_len):
    """创建下三角掩码"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    return mask

# 在 Attention 中应用:
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
scores = scores.masked_fill(mask == 0, float('-inf'))  # 未来的位置填 -inf
attn = F.softmax(scores, dim=-1)  # -inf → 0 attention weight
```

### 2.3 自回归生成 — 20min

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    """GPT 风格的自回归生成"""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    for _ in range(max_new_tokens):
        # 前向传播
        logits = model(input_ids)          # [B, T, vocab_size]
        next_logits = logits[:, -1, :]     # 只取最后一个位置
        next_logits = next_logits / temperature

        # 采样下一个 token
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 拼回去，继续
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0])
```

**自检**: 用这个 generate 生成 10 个 token，画出每一步 input_ids 的 shape 变化。

---

## 模块 3: LLM 训练三阶段（1 小时）

### 3.1 三阶段全景 — 20min

```
阶段 1: 预训练 (Pre-training)
  数据: 万亿 token 的互联网文本
  目标: Next Token Prediction（下一个词预测）
  成本: 数百万美元
  产出: Base Model（基础模型）

阶段 2: 监督微调 (SFT - Supervised Fine-Tuning)
  数据: 数万条高质量 QA/指令数据
  目标: 学会"对话格式"和"遵循指令"
  成本: 几千美元
  产出: Chat Model（对话模型）

阶段 3: 人类反馈强化学习 (RLHF)
  数据: 人类偏好标注
  目标: 对齐人类价值观
  成本: 数万美元
  产出: Aligned Model（对齐模型）
```

### 3.2 VLA 的训练映射 — 20min

```
对应到 VLA 训练:

预训练 ↗ 在大规模 VLM 基础上训练 (如 Llama + 通用图像)
SFT   ↗ 在机器人数据集上微调 (Open X-Embodiment)
RLHF  ↗ 用人类偏好优化机器人行为 (还比较早期)

目前 VLA 主流方案:
  1. 取一个预训练好的 VLM (视觉语言模型)
  2. 在机器人数据上 SFT
  3. 加入动作输出头，端到端微调
```

### 3.3 损失函数 — 10min

```python
# LLM 的训练损失: Cross-Entropy on Next Token
def compute_lm_loss(logits, labels):
    """
    logits: [B, T, vocab_size]  模型预测
    labels: [B, T]              真实 token
    """
    # shift: 用第 t 个 token 的输出去预测第 t+1 个 token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100  # 忽略 padding 位置
    )
    return loss
```

---

## 模块 4: LoRA 与 PEFT 高效微调（1.5 小时）

> ⭐ VLA 面试必问！这是你以后微调 VLA 模型的核心工具。

### 4.1 为什么需要 LoRA？ — 20min

```
全量微调 7B 参数模型:
  需要显存: ~56GB (模型参数 + 优化器状态 + 激活)
  训练时间: 数小时到数天
  
LoRA (Low-Rank Adaptation):
  只训练新增的小矩阵（~0.1% 参数）
  需要显存: ~16GB
  训练时间: 数十分钟到数小时

核心思想: ΔW ≈ B × A  (低秩分解)
  W 是 4096×4096 = 16M 参数 → 冻结
  B 是 4096×16 + A 是 16×4096 = 131K 参数 → 只训练这些
```

### 4.2 LoRA 数学 — 20min

```python
# 原始前向: h = W₀x
# LoRA 前向:  h = W₀x + BAx

# B: [d_out, r]  ← r 是秩 (rank)，一般 8~64
# A: [r, d_in]

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=16, lora_alpha=16):
        super().__init__()
        # 原始权重（冻结）
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA 参数（可训练）
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.scaling = lora_alpha / r  # LoRA 论文中的 α/r

    def forward(self, x):
        # 原始输出 + LoRA 修正
        return F.linear(x, self.weight) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)
```

### 4.3 PEFT 实战 — 30min

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# LoRA 配置 ← 这是你以后每次微调都要写的
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # 秩
    lora_alpha=32,           # 缩放系数
    lora_dropout=0.1,
    target_modules=[         # 对哪些层加 LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention 的 QKV
        "gate_proj", "up_proj", "down_proj",      # FFN
    ],
    bias="none",
)

# 包装模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: 8,388,608 || all params: 6,746,968,064 || trainable%: 0.1243%

# 训练（和普通 PyTorch 一样！）
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
# ... 标准训练循环 ...

# 保存（只保存 LoRA 权重，很小！）
model.save_pretrained("./lora_checkpoint")
# 只有 ~30MB，而不是 14GB！
```

### 4.4 QLoRA — 10min

```python
# 进一步节省显存：4-bit 量化 + LoRA
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,  # 加上这个就行
    device_map="auto",
)
# 7B 模型只用 ~4GB 显存！
```

**自检**: 如果让你用 LoRA 微调 OpenVLA，你会对哪些层加 LoRA？

---

## 模块 5: HuggingFace Transformers 实战（1.5 小时）

### 5.1 Pipeline 快速体验 — 20min

```python
from transformers import pipeline

# 文本生成
generator = pipeline("text-generation", model="gpt2")
print(generator("The robot picked up the", max_length=30))

# 这背后做的事情和 OpenVLA 调用 VLM 是一样的模式:
# Tokenize → Model Forward → Decode
```

### 5.2 理解模型加载 — 30min

```python
from transformers import AutoModel, AutoConfig

# 1. 查看模型配置（面试常考！）
config = AutoConfig.from_pretrained("gpt2")
print(f"vocab_size: {config.vocab_size}")   # 50257
print(f"n_layer: {config.n_layer}")         # 12
print(f"n_head: {config.n_head}")           # 12
print(f"n_embd: {config.n_embd}")           # 768

# 2. 修改配置建模型
my_config = AutoConfig.from_pretrained("gpt2")
my_config.n_layer = 4  # 更小的模型
small_model = AutoModel.from_config(my_config)

# 3. 冻结/解冻参数
for name, param in model.named_parameters():
    if 'lm_head' in name:     # 只训练输出头
        param.requires_grad = True
    else:
        param.requires_grad = False
```

### 5.3 实战: 微调一个小模型 — 40min

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model

# 1. 准备数据（用机器人指令数据举例）
data = {
    "text": [
        "<image> Pick up the red block </action>",
        "<image> Move the arm to the left </action>",
        "<image> Open the drawer </action>",
        # ... 实际需要几千条这样的数据
    ]
}
dataset = Dataset.from_dict(data)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
dataset = dataset.map(tokenize, batched=True)

# 2. 加载模型 + LoRA
model = AutoModelForCausalLM.from_pretrained("gpt2")
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], bias="none")
model = get_peft_model(model, lora_config)

# 3. 训练
training_args = TrainingArguments(
    output_dir="./vla_lora",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,  # 半精度
)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

---

## 🎤 费曼挑战（1.5 小时）

### 任务 1: 讲给奶奶听 (20min)
用最通俗的语言解释（写下来或录音）:
- **"大语言模型是什么？"** → 像一个读了很多书的大脑，你问它什么它能接着往下说
- **"什么是微调？"** → 像一个已经会开车的人，只需要学开卡车（不用重新学交规）
- **"LoRA 是什么？"** → 像一个已经写好的大部头书，你只在页边贴便利贴补充内容

### 任务 2: Tokenization 全流程 (20min)
选一句话: "Move the robot arm 30 degrees left"

在纸上画出完整的 Tokenization 流程:
```
原始文本 → Tokenizer → Token IDs → Embedding → 向量
```
每个步骤标注具体的数值（用 GPT-2 tokenizer 实际跑一遍）

### 任务 3: 代码讲解 (25min)
打开你的 LoRA 微调代码，向一个"只懂 PyTorch 不懂 LLM"的人解释:
1. 为什么用 `get_peft_model` 包装后参数变少了？
2. `target_modules` 选什么？为什么选这些？
3. LoRA 的 `r=16` 是什么意思？变大变小有什么影响？

### 任务 4: 一句话总结 (15min)
**面试题模拟**: "请解释 Attention 中 causal mask 的作用，以及为什么 GPT 必须用它但 BERT 不需要"

---

## 📝 今日自检清单

- [ ] 我能用 tokenizer 编解码一段文本
- [ ] 我能解释 BPE 的基本原理
- [ ] 我知道 GPT 和 BERT 的架构差异
- [ ] 我能画出 causal mask
- [ ] 我理解 LLM 训练三阶段（预训练 → SFT → RLHF）
- [ ] 我能配置 LoRA 并解释 r 和 alpha 的含义
- [ ] 我成功用 LoRA 微调了一个小模型
- [ ] 我完成了费曼讲解

---

## 💡 今日核心概念速查

| 概念 | 一句话 | VLA 中的角色 |
|------|--------|-------------|
| Tokenizer | 文字→数字的转换器 | 把指令转为模型输入 |
| GPT | 从左到右生成的下一个词预测器 | VLA 的语言骨架 |
| Causal Mask | 阻止模型"偷看未来" | 动作生成也必须是因果的 |
| LoRA | 低秩矩阵微调，省 99% 参数 | 微调 VLA 的核心技术 |
| PEFT | 参数高效微调的统称 | 让你用一张 GPU 微调 7B VLA |

---

> [!info] 知识库关联
> - [[../../../01-基础理论/大语言模型与微调/大语言模型与微调|大语言模型与微调]] — 自回归 + Tokenizer + Scaling Law
> - [[../../../01-基础理论/大语言模型与微调/LoRA与参数高效微调|LoRA与PEFT详细讲解]] — LoRA/QLoRA/Adapter/DoRA
> - [[../../../01-基础理论/大语言模型与微调/QLoRA完整实现|QLoRA完整实现]] — 4-bit量化 + 训练脚本
>
> ✅ **完成打卡**: NLP/LLM 基础已掌握，明天进入视觉领域！
>
> 🔜 **明日预告**: 计算机视觉与 ViT — 理解模型如何"看"图像，多模态模型如何融合视觉和语言！
