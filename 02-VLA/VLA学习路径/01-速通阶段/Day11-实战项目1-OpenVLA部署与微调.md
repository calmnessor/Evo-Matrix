# 💻 Day 11: 实战项目 1 — OpenVLA 部署与微调（8 小时）

> **口号**: "把论文变成代码——亲手部署一个 VLA 模型！"  
> **目标**: 完成 OpenVLA 的部署、推理、以及 LoRA 微调  
> **为什么重要**: 这是面试能说的"做过"的核心项目经验

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | 环境完整搭建与验证 | 1h | [ ] |
| 2 | 单次推理全流程 | 1.5h | [ ] |
| 3 | 批量推理与结果分析 | 1.5h | [ ] |
| 4 | LoRA 微调实战 | 2h | [ ] |
| 5 | 微调结果评估 | 1h | [ ] |
| 6 | MuJoCo 仿真深入实操（⭐ Day12前置） | 1.5h | [ ] |
| 7 | 项目总结 | 0.5h | [ ] |

---

## 项目目标

> 完成以下任务即为项目成功：
> 1. ✅ 成功加载 OpenVLA-7B 模型
> 2. ✅ 对至少 3 张不同的图片进行推理，获得有意义的动作输出
> 3. ✅ 用 LoRA 对小数据集微调，观察 loss 下降
> 4. ✅ 对比微调前后的输出差异

---

## 模块 1: 环境完整搭建与验证（1 小时）

### 1.1 硬性检查 — 15min

```bash
# 检查 GPU
nvidia-smi
# 期望: 至少 8GB 显存（推荐 12GB+）

# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"
# 期望: True

python -c "import torch; print(torch.cuda.get_device_name(0))"
# 期望: NVIDIA GPU 名称

# 检查 Python
python --version
# 期望: 3.10.x
```

### 1.2 安装 OpenVLA — 20min

```bash
# 如果之前没装完，继续
cd /path/to/projects
git clone https://github.com/openvla/openvla.git
cd openvla

# 安装
pip install -e .

# 验证导入
python -c "
from prismatic import load
from prismatic.models.vlms import PrismaticVLM
print('OpenVLA imported successfully!')
"
```

### 1.3 下载模型权重 — 25min

```python
# download_model.py
# 放在 openvla/ 目录下运行

from huggingface_hub import snapshot_download
import os

# 创建模型目录
os.makedirs("./models/openvla-7b", exist_ok=True)

# 下载（需要 huggingface token，去 huggingface.co 申请）
# 第一次可能需要登录: huggingface-cli login

print("Downloading OpenVLA-7B... (~15GB)")
snapshot_download(
    "openvla/openvla-7b",
    local_dir="./models/openvla-7b",
    ignore_patterns=["*.msgpack", "*.h5"],
    # token="hf_xxx",  # 如果需要
)
print("Download complete!")

# 验证文件
import glob
files = glob.glob("./models/openvla-7b/*")
print(f"Files downloaded: {len(files)}")
for f in files[:10]:
    print(f"  {f}")
```

---

## 模块 2: 单次推理全流程（1.5 小时）

### 2.1 最小推理脚本 — 30min

创建 `inference_demo.py`:

```python
"""OpenVLA 最小推理示例"""
import torch
import numpy as np
from PIL import Image
import argparse

# ===== 1. 加载模型和处理器 =====
print("Loading OpenVLA... (this may take 1-2 minutes)")
from transformers import AutoModelForVision2Seq, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,  # 半精度
    device_map="auto",
    trust_remote_code=True,
    # 如果显存不够，取消下面的注释:
    # load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
)
print("Model loaded!")

# ===== 2. 准备测试图像 =====
# 下载一张示例图或使用自己的图片
# 如果没有真实机器人图像，用网络图片测试也是可以的
import requests
from io import BytesIO

# 示例: 下载一张包含桌面物体的图片
# url = "https://example.com/your-robot-view.jpg"
# response = requests.get(url)
# image = Image.open(BytesIO(response.content)).convert("RGB")

# 或者创建一个测试图像
image = Image.new('RGB', (384, 384), color='white')
# 在实际使用中替换为真实图像: image = Image.open("robot_view.jpg").convert("RGB")
print(f"Input image size: {image.size}")

# ===== 3. 准备指令 =====
instruction = "pick up the red block"
print(f"Instruction: {instruction}")

# ===== 4. 推理 =====
print("Running inference...")
inputs = processor(
    images=image,
    text=instruction,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

# ===== 5. 解码 =====
output_text = processor.decode(outputs[0], skip_special_tokens=True)
print(f"\n{'='*50}")
print(f"Raw output: {output_text}")
print(f"{'='*50}")

# 解析 action 值
def parse_action(output_text):
    """
    解析 OpenVLA 输出的动作值
    注意: 输出格式取决于模型版本和训练方式
    """
    # 尝试多种解析方式

    # 方式 1: 直接找数字
    import re
    numbers = re.findall(r'-?\d+\.?\d*', output_text)
    if len(numbers) >= 7:
        return [float(n) for n in numbers[:7]]

    # 方式 2: 找特殊标记
    if '<ACT>' in output_text:
        act_part = output_text.split('<ACT>')[1].split('</ACT>')[0]
        tokens = act_part.strip().split()
        # 反量化动作
        action = []
        for t in tokens:
            try:
                # 假设 token 是 bin index
                bin_val = int(t)
                continuous = (bin_val / 127.5) - 1.0
                action.append(continuous)
            except:
                pass
        if len(action) >= 7:
            return action[:7]

    return None

action = parse_action(output_text)
if action:
    print(f"\nParsed action (7-DOF):")
    labels = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']
    for label, val in zip(labels, action):
        print(f"  {label:10s}: {val:+.4f}")
else:
    print("\nCould not parse action from output.")
    print("This is expected for the first run without real data.")
    print("Try with a real robot image or check the model version.")

print(f"\nInference complete!")
```

### 2.2 测试用不同指令 — 30min

```python
# 对同一张图，试不同指令，观察输出变化
instructions = [
    "pick up the red block",
    "move the arm to the left",
    "open the drawer",
    "place the object on the table",
    "push the button",
]

# 对每个指令运行推理
for inst in instructions:
    print(f"\n{'='*50}")
    print(f"Instruction: {inst}")
    # ... 使用上面的推理代码 ...
    print(f"Output length: {len(output_text)} chars")
```

### 2.3 常见问题排查 — 30min

```python
# 问题 1: CUDA Out of Memory
# 解决:
# a) 4-bit 量化
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=quant_config,
    device_map="auto",
)

# b) CPU offload
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True,
)

# 问题 2: 输出全一样
# 原因: 可能是 do_sample=False 但模型本身不确定性来源不够
# 解决: 试着 do_sample=True, temperature=0.7

# 问题 3: 输出没有动作值
# 原因: 可能 model 版本不同，输出格式不同
# 解决: 检查 model.config 中的 action_config
print(model.config)
```

---

## 模块 3: 批量推理与结果分析（1.5 小时）

### 3.1 批量推理脚本 — 40min

```python
"""批量推理脚本"""
import json
import os
from pathlib import Path

def batch_inference(model, processor, image_dir, instructions_file, output_dir):
    """
    批量推理:
      image_dir: 图片目录
      instructions_file: JSON 文件，{image_name: instruction}
      output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取指令
    with open(instructions_file) as f:
        instructions = json.load(f)

    results = {}
    for img_name, inst in instructions.items():
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"SKIP: {img_path} not found")
            continue

        print(f"\nProcessing: {img_name}")
        print(f"  Instruction: {inst}")

        # 加载图片
        image = Image.open(img_path).convert("RGB")

        # 推理
        inputs = processor(images=image, text=inst, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        output_text = processor.decode(outputs[0], skip_special_tokens=True)
        action = parse_action(output_text)

        results[img_name] = {
            "instruction": inst,
            "raw_output": output_text,
            "action": action,
        }

        print(f"  Action: {action}")

    # 保存结果
    output_path = os.path.join(output_dir, "inference_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return results

# 创建示例指令文件
sample_instructions = {
    "img_001.jpg": "pick up the red cup",
    "img_002.jpg": "open the drawer",
    "img_003.jpg": "move the block to the right",
}
with open("instructions.json", "w") as f:
    json.dump(sample_instructions, f, indent=2)

# 运行
# results = batch_inference(model, processor, "./images", "instructions.json", "./outputs")
```

### 3.2 结果分析 — 30min

```python
"""分析推理结果"""
import matplotlib.pyplot as plt
import numpy as np

def analyze_results(results):
    """可视化分析批量推理结果"""
    # 提取所有动作
    all_actions = []
    labels = []

    for img_name, data in results.items():
        if data['action']:
            all_actions.append(data['action'])
            labels.append(img_name)

    if not all_actions:
        print("No valid actions to analyze")
        return

    # 转为 numpy
    actions = np.array(all_actions)

    # 绘制每个维度的分布
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    dim_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']

    for i, (ax, name) in enumerate(zip(axes.flatten()[:7], dim_names)):
        ax.bar(range(len(actions)), actions[:, i])
        ax.set_title(f'Action Dimension: {name}')
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    axes.flatten()[7].axis('off')
    plt.suptitle('OpenVLA Action Outputs Across Images')
    plt.tight_layout()
    plt.savefig('action_analysis.png', dpi=150)
    print("Analysis plot saved to action_analysis.png")
```

### 3.3 理解输出质量 — 20min

```
判断推理结果是否合理:

✅ 好的信号:
  - 动作值在 [-1, 1] 范围内
  - 不同指令产生不同的动作
  - gripper 值对应任务（抓取 = 1, 放下 = 0）
  - 动作幅度合理（不是全部接近 0）

❌ 坏的信号:
  - 所有指令输出都相同
  - 动作值异常大（>2 或 <-2）
  - 输出完全是乱码
  - 所有动作为 0

如果输出不好:
  1. 检查输入图像的分辨率是否匹配模型期望
  2. 检查图像是否做了正确的归一化
  3. 可能需要用真实机器人数据微调
```

---

## 模块 4: LoRA 微调实战（2 小时）

### 4.1 准备微调数据 — 30min

```python
"""准备微调数据
即使没有真实机器人数据，也可以创建合成数据来练习流程
"""
import json
import numpy as np

def create_synthetic_training_data(n_samples=100):
    """创建合成的训练数据"""
    data = []
    for i in range(n_samples):
        # 随机位置的目标
        target_x = np.random.uniform(-0.5, 0.5)
        target_y = np.random.uniform(-0.5, 0.5)
        target_z = np.random.uniform(0.0, 0.3)

        # 构造指令
        instructions = [
            f"move to position x={target_x:.2f} y={target_y:.2f}",
            f"pick up the object at ({target_x:.2f}, {target_y:.2f})",
            f"reach the target location",
        ]
        instruction = np.random.choice(instructions)

        # 构造"专家"动作
        # (实际上这是一个简单的映射，真实情况需要遥操作数据)
        action = [
            np.clip(target_x, -1, 1),
            np.clip(target_y, -1, 1),
            np.clip(target_z, -1, 1),
            np.random.uniform(-0.1, 0.1),  # roll
            np.random.uniform(-0.1, 0.1),  # pitch
            np.random.uniform(-0.1, 0.1),  # yaw
            1.0,  # gripper: close to grasp
        ]

        data.append({
            "instruction": instruction,
            "action": action,
            # 实际训练需要图片路径
            # "image": f"images/train_{i:04d}.jpg"
        })

    # 保存
    with open("synthetic_training_data.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Created {n_samples} synthetic training samples")
    return data

create_synthetic_training_data(100)
```

### 4.2 LoRA 微调代码 — 60min

```python
"""OpenVLA LoRA 微调脚本"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from PIL import Image
import json
import os

# ===== 1. 加载基模型 =====
print("Loading base model...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
)

# ===== 2. 配置 LoRA =====
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                # LoRA rank
    lora_alpha=32,       # 缩放系数
    lora_dropout=0.1,
    target_modules=[
        # Llama-2 的 Attention + FFN 层
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 期望: ~8M 可训练参数 / ~7B 总参数 = ~0.1%

# ===== 3. 准备数据集 =====
def load_dataset(data_path, image_dir):
    """加载微调数据"""
    with open(data_path) as f:
        raw_data = json.load(f)

    formatted_data = []
    for item in raw_data:
        # 原始格式可能是:
        # {"image": "xxx.jpg", "instruction": "pick up...", "action": [...]}
        formatted_data.append({
            "image": os.path.join(image_dir, item.get("image", "")),
            "text": item["instruction"],
            # 动作需要转换成 token 格式（取决于 OpenVLA 版本）
        })

    return Dataset.from_list(formatted_data)

# 如有真实数据:
# dataset = load_dataset("training_data.json", "./images")

# 使用占位数据做演示:
dummy_data = [
    {"image": None, "text": "pick up the red block"},
    {"image": None, "text": "move the arm left"},
    {"image": None, "text": "open the drawer"},
]
dataset = Dataset.from_list(dummy_data)

# ===== 4. Tokenize =====
def tokenize_function(examples):
    """注意: 这只是文本 tokenize，实际 VLA 需要同时处理图像"""
    return processor.tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ===== 5. 训练 =====
training_args = TrainingArguments(
    output_dir="./openvla_lora_output",
    per_device_train_batch_size=2,    # 根据显存调整
    gradient_accumulation_steps=4,     # 模拟更大 batch
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,                        # 用 bf16
    report_to="none",                 # 不上报 wandb
    # 关键: 只保存 LoRA 权重
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("\n" + "="*50)
print("Starting LoRA fine-tuning...")
print("="*50)

trainer.train()

# ===== 6. 保存 LoRA 权重 =====
lora_save_path = "./openvla_lora_checkpoint"
model.save_pretrained(lora_save_path)
processor.save_pretrained(lora_save_path)

print(f"\nLoRA weights saved to: {lora_save_path}")
print(f"Checkpoint size: ~30MB (vs ~14GB for full model)")

# ===== 7. 加载微调后的模型进行推理 =====
print("\nLoading fine-tuned model for inference...")
from peft import PeftModel

# 重新加载基模型
base_model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, lora_save_path)
model = model.merge_and_unload()  # 合并 LoRA 到基模型（可选）
print("Fine-tuned model ready!")
```

### 4.3 微调技巧 — 20min

```
关键调整参数与经验:

r (LoRA rank):
  - r=8: 极省参数，效果还行
  - r=16: 平衡点 ← 推荐
  - r=64: 效果好但参数多

learning_rate:
  - 1e-4: 较稳定
  - 2e-4: 收敛快（推荐）
  - 5e-4: 可能不稳定

batch_size:
  - 尽量大！但受限于显存
  - 实际 effective batch = batch_size × gradient_accumulation_steps
  - 目标: 32-64

训练 epoch:
  - 1-3 epochs: 通常足够（多了过拟合）
  - VLA 数据量大时 1 epoch 即可

常见坑:
  1. 忘记冻结视觉 backbone → 显存爆
  2. 数据没归一化 → 训练不稳定
  3. 不同数据集 action space 不同 → 需要统一
```

---

## 模块 5: 微调结果评估（1 小时）

### 5.1 快速评估 — 30min

```python
def evaluate_model(model, processor, test_cases):
    """
    评估微调前后的模型
    test_cases: list of (image_path, instruction, expected_action)
    """
    results = []
    for img_path, instruction, expected in test_cases:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=instruction,
                          return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)

        output_text = processor.decode(outputs[0])
        pred_action = parse_action(output_text)

        # 如果有预期动作，计算误差
        error = None
        if expected and pred_action:
            error = np.mean((np.array(pred_action) - np.array(expected))**2)

        results.append({
            "instruction": instruction,
            "predicted": pred_action,
            "expected": expected,
            "mse": error,
        })

    # 打印汇总
    valid_results = [r for r in results if r['mse'] is not None]
    if valid_results:
        avg_mse = np.mean([r['mse'] for r in valid_results])
        print(f"Average MSE: {avg_mse:.6f}")

    return results

# 创建简单的测试用例
test_cases = [
    ("test_001.jpg", "pick up the red block", None),
    ("test_002.jpg", "move to the left", None),
    ("test_003.jpg", "open the drawer", None),
]
# results = evaluate_model(model, processor, test_cases)
```

### 5.2 对比分析 — 30min

```python
def compare_before_after(base_model, fine_tuned_model, processor, test_cases):
    """对比微调前后模型的输出"""
    print(f"{'Instruction':<30} {'Base':<20} {'Fine-tuned':<20}")
    print("-" * 70)

    for img_path, instruction, _ in test_cases:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=instruction,
                          return_tensors="pt").to(base_model.device)

        # 基模型
        with torch.no_grad():
            base_out = base_model.generate(**inputs, max_new_tokens=256)
        base_text = processor.decode(base_out[0])[:50]

        # 微调模型
        with torch.no_grad():
            ft_out = fine_tuned_model.generate(**inputs, max_new_tokens=256)
        ft_text = processor.decode(ft_out[0])[:50]

        print(f"{instruction:<30} {base_text:<20} {ft_text:<20}")

# compare_before_after(base_model, fine_tuned_model, processor, test_cases)
```

---

## 模块 6: MuJoCo 仿真环境深入实操（1.5 小时）

> ⭐ 这是 Day 12 ACT 训练的前置技能——学会建仿真环境，才能训练策略。

### 6.1 安装与验证 — 15min

```bash
pip install mujoco
pip install mujoco-python-viewer  # 可视化

python -c "
import mujoco
print(f'MuJoCo version: {mujoco.__version__}')
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
print('MuJoCo OK')
"
```

### 6.2 从零构建一个机器人仿真场景 — 40min

```xml
<!-- pick_and_place.xml — 桌面取放场景 -->
<mujoco model="pick_and_place">
  <compiler angle="degree"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <!-- 桌面纹理 -->
    <texture name="table_tex" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.8 0.8 0.8" rgb2="0.6 0.6 0.6"/>
    <material name="table_mat" texture="table_tex" texrepeat="4 4" reflectance="0.2"/>
  </asset>

  <worldbody>
    <!-- 光照 -->
    <light name="top" pos="0 0 2" directional="false"/>

    <!-- 桌面 -->
    <geom name="table" type="box" pos="0.5 0 -0.05" size="0.6 0.6 0.05"
          material="table_mat"/>

    <!-- 目标物体（红色方块） -->
    <body name="target_object" pos="0.5 0.2 0.03">
      <freejoint/>  <!-- 自由物体，可以被推动 -->
      <geom name="object_geom" type="box" size="0.03 0.03 0.03"
            rgba="1 0.2 0.2 1" mass="0.1"/>
    </body>

    <!-- 机械臂基座 -->
    <body name="arm_base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.08 0.05" rgba="0.3 0.3 0.3 1"/>

      <!-- === 肩部关节 (Shoulder) === -->
      <joint name="shoulder_pan"  type="hinge" axis="0 0 1"   range="-180 180"/>
      <body name="shoulder" pos="0 0 0.05">
        <geom name="shoulder_geom" type="sphere" size="0.06" rgba="0.7 0.7 0.7 1"/>

        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-130 130"/>
        <body name="upper_arm" pos="0 0 0">
          <geom name="upper_arm_geom" type="capsule" fromto="0 0 0 0.3 0 0"
                size="0.04" rgba="0.5 0.5 0.8 1"/>

          <!-- === 肘部关节 (Elbow) === -->
          <joint name="elbow" type="hinge" axis="0 1 0" range="-130 130"/>
          <body name="forearm" pos="0.3 0 0">
            <geom name="forearm_geom" type="capsule" fromto="0 0 0 0.25 0 0"
                  size="0.035" rgba="0.5 0.7 0.5 1"/>

            <!-- === 腕部关节 (Wrist) === -->
            <joint name="wrist_pitch" type="hinge" axis="0 1 0" range="-90 90"/>
            <joint name="wrist_roll"  type="hinge" axis="1 0 0" range="-360 360"/>
            <body name="wrist" pos="0.25 0 0">
              <geom name="wrist_geom" type="sphere" size="0.04" rgba="0.8 0.8 0.3 1"/>

              <!-- === 夹爪 (Gripper) === -->
              <body name="gripper_base" pos="0.05 0 0">
                <!-- 夹爪驱动通过 tendon 实现 -->
                <geom name="finger_left"  type="box" pos="0.03 0.03 0"
                      size="0.04 0.005 0.015" rgba="0.3 0.3 0.3 1"/>
                <geom name="finger_right" type="box" pos="0.03 -0.03 0"
                      size="0.04 0.005 0.015" rgba="0.3 0.3 0.3 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- 相机（渲染用） -->
    <camera name="front" pos="1.0 0 0.8" xyaxes="0 1 0 -0.8 0 1"
            fovy="50" resolution="640 480"/>
  </worldbody>

  <!-- === 执行器配置 === -->
  <actuator>
    <motor name="m_shoulder_pan"  joint="shoulder_pan"  gear="50" ctrlrange="-2 2"/>
    <motor name="m_shoulder_lift" joint="shoulder_lift" gear="50" ctrlrange="-2 2"/>
    <motor name="m_elbow"         joint="elbow"         gear="50" ctrlrange="-2 2"/>
    <motor name="m_wrist_pitch"   joint="wrist_pitch"   gear="30" ctrlrange="-2 2"/>
    <motor name="m_wrist_roll"    joint="wrist_roll"    gear="30" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
```

```python
# 加载并运行自定义场景
import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("pick_and_place.xml")
data = mujoco.MjData(model)

# 设置初始关节角度
init_qpos = np.array([0, -30, -60, 0, 0])  # degrees
data.qpos[:] = np.radians(init_qpos)
mujoco.mj_forward(model, data)

# 使用 viewer 查看
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -30

    for step in range(2000):
        # 简单的正弦控制（VLA 会替代这一步！）
        t = step * model.opt.timestep
        data.ctrl[0] = 0.3 * np.sin(2 * t)     # shoulder 摆动
        data.ctrl[1] = -0.5                      # shoulder lift 下压
        data.ctrl[2] = -0.3 + 0.2 * np.sin(t)   # elbow 微动

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
```

### 6.3 MuJoCo 物理参数调优 — 25min

> VLA 仿真中最容易被忽略的是物理参数，参数不对 → sim2real 跨

```xml
<!-- 物理参数配置示例 -->
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81">
    <!-- 接触求解器 -->
    <flag contact="enable" gravity="enable"/>
  </option>

  <!-- === 全局默认物理参数 === -->
  <default>
    <!-- 关节默认 -->
    <joint damping="0.5"          <!-- 阻尼 → 关节速度不会无限增长 -->
           frictionloss="0.1"     <!-- 摩擦力损失 -->
           armature="0.01"/>      <!-- 电机电枢惯量 -->

    <!-- 几何体默认 -->
    <geom friction="1.0 0.005 0.0001"  <!-- 滑动/扭转/滚动摩擦 -->
          solref="0.02 1"              <!-- 接触求解器参数 -->
          solimp="0.9 0.95 0.001"/>    <!-- 接触阻抗参数 -->

    <!-- 电机默认 -->
    <motor forcerange="-5 5"      <!-- 力/力矩范围 -->
           ctrlrange="-2 2"/>     <!-- 控制范围 -->
  </default>
</mujoco>
```

```python
# 关键物理参数速查

# 1. 摩擦参数 (用于 geom)
# geom friction = [sliding, torsional, rolling]
# 例: friction="1.0 0.005 0.0001"
#   sliding=1.0:   滑动摩擦系数（金属≈0.1-0.3, 橡胶≈0.7-1.0）
#   torsional=0.005: 扭转摩擦
#   rolling=0.0001:  滚动摩擦

# 2. 接触求解器参数 (solref)
# solref = [timeconst, dampratio]
#   timeconst: 时间常数, 越小→响应越快→越不稳定
#   dampratio: 阻尼比, 1=临界阻尼
#   典型值: "0.02 1" (较快), "0.01 0.9" (更快但可能震荡)

# 3. 时间步 (timestep)
# 物理仿真用 0.001-0.005s
# 控制用 0.01-0.05s (每 N 个物理步发一次动作)
#   例: timestep=0.002, 控制频率=20Hz → 每 25 步发一次动作

# 4. 质量与惯量
# 质量太小 → 物体飞出（Havok 物理）
# 质量太大 → 需要大力矩, 可能超出 motor forcerange
# 建议: 参考真实机器人参数
```

### 6.4 与 Python 控制的完整对接 — 20min

```python
"""完整的 MuJoCo 仿真控制循环 — VLA 策略将替代中间的 controller"""

class MuJoCoSimEnv:
    """VLA 训练/测试用的仿真环境封装"""
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.n_actions = self.model.nu  # 执行器数量
        self.n_obs = self.model.nq + self.model.nv  # 位置+速度

        # 相机配置
        self.renderer = mujoco.Renderer(self.model, 480, 640)

    def reset(self, qpos_init=None):
        """重置环境"""
        if qpos_init is not None:
            self.data.qpos[:] = qpos_init
        else:
            # 用 keyframe 或随机初始化
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, action):
        """执行一步动作
        action: [n_actions] 范围 [-1, 1]
        """
        # 反归一化到电机控制范围
        ctrl_min = self.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.model.actuator_ctrlrange[:, 1]
        ctrl = ctrl_min + (action + 1) / 2 * (ctrl_max - ctrl_min)

        # 执行
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

        return self.get_obs()

    def get_obs(self):
        """获取观测"""
        # 关节位置 + 速度
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # 渲染图像
        self.renderer.update_scene(self.data)
        img = self.renderer.render()

        return {
            "qpos": qpos,          # 关节位置 [nq]
            "qvel": qvel,          # 关节速度 [nv]
            "image": img,          # RGB 图像 [480, 640, 3]
            "ee_pos": self.data.site_xpos[0].copy(),  # 末端位置
        }

    def get_reward(self):
        """计算奖励（任务相关）"""
        ee_pos = self.data.site_xpos[0]    # 末端位置
        obj_pos = self.data.body("target_object").xpos  # 物体位置
        distance = np.linalg.norm(ee_pos - obj_pos)
        return -distance  # 负距离 = 越近越好

# 使用
env = MuJoCoSimEnv("pick_and_place.xml")
obs = env.reset(np.radians([0, -30, -60, 0, 0]))

for step in range(1000):
    # 这里就是 VLA 策略的位置！
    action = np.random.uniform(-1, 1, env.n_actions)  # 随机（演示用）

    obs = env.step(action)

    if step % 100 == 0:
        reward = env.get_reward()
        print(f"Step {step:4d}: reward={reward:.4f}")

# 保存渲染图像（模拟 VLA 的输入数据）
import imageio
imageio.imwrite("sim_obs.png", obs["image"])
print("Saved sim_obs.png")
```

---

## 📋 项目总结（1 小时）

### 今日产出清单

完成以下全部即项目成功:

- [ ] 成功安装 OpenVLA 环境
- [ ] 成功下载并加载 OpenVLA-7B 模型
- [ ] 至少完成 3 次单图推理
- [ ] 写好了批量推理脚本
- [ ] 理解了 LoRA 微调流程
- [ ] 成功运行了 LoRA 微调（即使是合成数据）
- [ ] 保存了 LoRA checkpoint
- [ ] 比较了基模型和微调模型的输出差异
- [ ] 成功加载自定义 MuJoCo 场景（pick_and_place.xml）
- [ ] 跑了完整的仿真控制循环，输出渲染图像

### 面试可以这么说

```
"我做过 OpenVLA 的部署和微调:
 - 使用 HuggingFace transformers 加载 7B 模型
 - 理解其 SigLIP + DINOv2 + Llama-2 的架构
 - 用 LoRA 对动作输出进行了微调
 - 了解了 visual token 和 text token 的拼接方式
 - 掌握了 4-bit 量化和混合精度推理技巧
 
 我也搭建过 MuJoCo 仿真环境:
 - 用 MJCF XML 从零定义了机械臂和桌面场景
 - 理解执行器、传感器、物理参数的配置
 - 封装了标准的 step/reset/render 接口（适配 VLA 策略）"
```

### 遇到的坑记录（自己填）

```
1. _________________________________
2. _________________________________
3. _________________________________
```

---

> ✅ **完成打卡**: 第一个 VLA 项目 + MuJoCo 仿真完成！明天 ACT 训练！
>
> 🔜 **明日预告**: Day 12 — 在 MuJoCo 仿真环境中训练 ACT 策略！
