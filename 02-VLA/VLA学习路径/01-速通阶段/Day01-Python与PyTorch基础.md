# 🚀 Day 1: Python 进阶 + PyTorch 基础速通（8 小时）

> **口号**: "代码是具身智能的骨架，今天把骨架搭好！"  
> **目标**: 达到能流畅使用 PyTorch 写模型代码的水平  
> **费曼提醒**: 每学完一个模块，立刻用自己的话讲一遍

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | Python 进阶特性复习 | 1.5h | [ ] |
| 2 | NumPy 核心操作 | 1h | [ ] |
| 3 | PyTorch Tensor 基础 | 1.5h | [ ] |
| 4 | PyTorch Autograd 自动求导 | 1.5h | [ ] |
| 5 | PyTorch nn.Module 与 Dataset | 1.5h | [ ] |
| 6 | 费曼挑战 + 总结输出 | 1h | [ ] |

---

## 模块 1: Python 进阶特性（1.5 小时）

> VLA 代码中大量使用以下特性，必须熟练掌握。

### 1.1 类型注解 (Type Hints) — 15min

VLA 项目代码量大，类型注解是理解接口的关键。

```python
from typing import List, Dict, Optional, Union, Tuple
import torch

# 常见的 VLA 代码中的类型注解
def encode_image(
    image: torch.Tensor,                    # [B, C, H, W]
    model: torch.nn.Module,
) -> torch.Tensor:                          # [B, D] 特征向量
    """将图像编码为特征向量"""
    return model(image)

# 复杂的嵌套类型
def process_batch(
    observations: Dict[str, torch.Tensor],  # {"image": ..., "state": ...}
    language_instruction: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ...
```

**自检题**: 写一个函数签名，输入是一个 batch 的观测（包含图像和关节状态），输出是动作序列。

### 1.2 Dataclass 数据类 — 15min

VLA 中配置、数据样本大量使用 dataclass。

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class RobotTrajectory:
    """一个机器人轨迹数据"""
    images: List[torch.Tensor]      # 图像序列
    actions: torch.Tensor           # [T, action_dim]
    instruction: str                # 语言指令
    timestamps: List[float] = field(default_factory=list)

# 这就是 OpenVLA 等项目中数据组织的基本单元
```

### 1.3 上下文管理器与装饰器 — 20min

```python
# VLA 推理中几乎必用
@torch.no_grad()  # 禁用梯度计算，节省显存
def inference(model, image, instruction):
    ...

# 混合精度推理
with torch.cuda.amp.autocast():
    output = model(image, instruction)
```

### 1.4 生成器与迭代器 — 10min

```python
# 处理大型机器人数据集时必用
def trajectory_generator(data_dir):
    """生成器模式：逐条产出轨迹，不一次性加载到内存"""
    for file in sorted(os.listdir(data_dir)):
        yield torch.load(os.path.join(data_dir, file))
```

### 1.5 torch.einsum 预告 — 10min

`einsum` 是多头注意力的核心写法：

```python
# 注意力计算: Q @ K^T
# scores[b, h, s, s] = Q[b, h, s, d] @ K[b, h, s, d].T
scores = torch.einsum('b h s d, b h t d -> b h s t', Q, K)
```

**快速自检**: 能否看懂 `'b h s d, b h t d -> b h s t'` 的含义？

### 📖 推荐阅读
- Python 官方 typing 文档（15 分钟浏览）
- Real Python: `dataclasses` tutorial

---

## 模块 2: NumPy 核心操作（1 小时）

> 重点: 广播机制、花式索引、向量化思维

### 2.1 广播机制 (Broadcasting) — 20min

这是 VLA 代码中最容易出 Bug 的地方。

```python
import numpy as np

# 规则: 从最后一维开始对齐，维度为1的可以广播
# 例子: batch 图像归一化
images = np.random.randn(32, 3, 224, 224)  # [B, C, H, W]
mean = np.array([0.485, 0.456, 0.406])     # [C]
std = np.array([0.229, 0.224, 0.225])      # [C]

# 广播: [C] → [1, C, 1, 1] → 自动扩展到 [32, 3, 224, 224]
normalized = (images - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
```

**自检题**: `[32, 1, 224, 224]` 和 `[3, 224, 224]` 能否广播？能的话结果 shape 是什么？

### 2.2 花式索引 (Fancy Indexing) — 20min

```python
# 取 batch 中特定索引的数据
indices = [0, 5, 10]
batch_data[indices]  # shape: [3, ...]

# 布尔索引
mask = actions[:, 0] > 0.5  # 取第一个关节角度大于0.5的动作
filtered_actions = actions[mask]
```

### 2.3 向量化思维 — 20min

```python
# ❌ 慢: 循环处理
results = []
for i in range(len(trajectories)):
    results.append(process(trajectories[i]))

# ✅ 快: 向量化
results = np.stack([t.actions for t in trajectories])  # 一次性处理
```

---

## 模块 3: PyTorch Tensor 基础（1.5 小时）

### 3.1 Tensor 创建与类型 — 20min

```python
import torch

# 从 NumPy 来（VLA 数据集常用）
arr = np.random.randn(10, 3)
tensor = torch.from_numpy(arr).float()  # float32

# 直接创建
x = torch.randn(2, 3, 224, 224)    # 批图像
y = torch.zeros(4, 7)              # 7 自由度动作

# 设备管理 — VLA 核心
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
images = images.to(device)
```

### 3.2 Tensor 操作 — 30min

```python
# 变形: 图像 patchify（ViT 的关键操作）
# [B, C, H, W] → [B, num_patches, patch_dim]
x = torch.randn(2, 3, 224, 224)
patches = x.unfold(2, 16, 16).unfold(3, 16, 16)  # [2, 3, 14, 14, 16, 16]
patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, 196, 768)

# 拼接: 多模态特征融合常用
vision_feat = torch.randn(4, 512)
lang_feat = torch.randn(4, 768)
fused = torch.cat([vision_feat, lang_feat], dim=-1)  # [4, 1280]

# 广播规则与 NumPy 一致
```

### 3.3 DataLoader 基础 — 20min

```python
from torch.utils.data import Dataset, DataLoader

class RobotDataset(Dataset):
    """最简单的机器人数据集"""
    def __init__(self, data_dir):
        self.files = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return {
            'image': data['image'],      # [C, H, W]
            'action': data['action'],    # [action_dim]
            'instruction': data['text'], # str
        }

loader = DataLoader(RobotDataset('./data'), batch_size=32, shuffle=True)
```

**自检题**: 自己写一个 Dataset 类，每次返回 (image, text, action) 三元组。

### 3.4 GPU 显存管理 — 20min

```python
# 查看显存（VLA 模型大，显存管理是必备技能）
print(torch.cuda.memory_summary())

# 释放显存
del model
torch.cuda.empty_cache()

# 推理时关闭梯度
with torch.no_grad():
    output = model(input)
```

---

## 模块 4: Autograd 自动求导（1.5 小时）

> 这是理解"训练"的核心！

### 4.1 计算图原理 — 30min

```python
# PyTorch 自动记录所有操作
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2           # y = x²
z = 3 * y + 1        # z = 3x² + 1

z.backward()          # 计算 dz/dx
print(x.grad)         # dz/dx = 6x = 12 ✓

# 在脑子里走一遍: z = 3x²+1 → dz/dx = 6x → 当 x=2, dz/dx = 12
```

### 4.2 典型训练循环 — 30min

```python
# 这是所有 VLA 训练代码的核心骨架
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 前向传播
        pred_actions = model(batch['image'], batch['instruction'])
        loss = criterion(pred_actions, batch['actions'])

        # 2. 反向传播
        optimizer.zero_grad()   # 清零梯度（忘记上一步）
        loss.backward()          # 计算梯度

        # 3. 更新参数
        optimizer.step()         # 沿梯度下降

        # 4. 记录
        print(f"Loss: {loss.item():.4f}")
```

### 4.3 常见坑 — 20min

```python
# 坑 1: 忘记 zero_grad() → 梯度累积导致不收敛
# 坑 2: tensor 离开计算图后还想 backward → 用 .detach()
# 坑 3: eval() 模式下忘记 torch.no_grad()
loss = loss.detach().cpu().item()  # 记录时 detach
```

### 4.4 Gradient Checkpointing — 10min

大模型常见技巧：用时间换空间。

```python
# VLA 模型 7B 参数，不这样做放不下
from torch.utils.checkpoint import checkpoint
output = checkpoint(self.encoder, x)  # 不存中间激活，反向时重算
```

---

## 模块 5: nn.Module 与训练范式（1.5 小时）

### 5.1 模型构建模板 — 40min

```python
import torch.nn as nn

class VLAPolicy(nn.Module):
    """VLA 策略的简化骨架"""
    def __init__(self, vision_dim=512, lang_dim=768, action_dim=7):
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, vision_dim),
        )
        self.lang_proj = nn.Linear(lang_dim, 512)
        self.action_head = nn.Sequential(
            nn.Linear(vision_dim + 512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        # 注意: 这是极度简化的，真正的 VLA 用的是 Transformer

    def forward(self, image, lang_embedding):
        v = self.vision_encoder(image)      # [B, vision_dim]
        l = self.lang_proj(lang_embedding)  # [B, 512]
        fused = torch.cat([v, l], dim=-1)   # [B, vision_dim + 512]
        actions = self.action_head(fused)   # [B, action_dim]
        return actions
```

**自检题**: 修改上面的模型，使其支持多帧图像输入 `[B, T, C, H, W]`。

### 5.2 常见层速查 — 20min

```python
# 这些是 VLA 模型中出现频率最高的层
nn.Linear(in, out)         # 全连接
nn.LayerNorm(dim)          # Transformer 标配
nn.Conv2d(in_c, out_c, k)  # 图像特征提取
nn.MultiheadAttention      # 注意力
nn.Embedding(vocab, dim)   # Token 嵌入
nn.Dropout(p)              # 正则化
nn.GELU()                  # Transformer 激活函数（不是 ReLU!）
```

### 5.3 保存与加载 — 15min

```python
# 完整保存（含优化器状态，用于断点续训）
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pt')

# 只保存模型权重（用于部署推理）
torch.save(model.state_dict(), 'policy.pt')

# 加载
model.load_state_dict(torch.load('policy.pt'))
```

### 5.4 混合精度训练 — 15min

```python
# 7B 参数的 VLA，不开混合精度根本跑不动
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():  # 自动混合精度
    output = model(image, text)
    loss = criterion(output, target)

scaler.scale(loss).backward()    # 缩放防止下溢
scaler.step(optimizer)
scaler.update()
```

---

## 🎤 费曼挑战（1 小时）

### 任务 1: 口头讲解（30min）
对着一面墙/镜子，用中文讲解以下概念，**不能看笔记**：

1. **Autograd 的工作原理**: 什么是计算图？`backward()` 做了什么？
2. **训练循环的五要素**: `forward → loss → zero_grad → backward → step`
3. **为什么 VLA 模型需要混合精度训练？**

录下来听一遍，哪里卡住了就回去复习。

### 任务 2: 代码输出（20min）
从零写出以下代码 **（不要复制粘贴）**:

```python
# 要求: 写一个完整的训练脚本，包含:
# 1. 一个简单的 MLP 模型
# 2. 随机生成的伪数据
# 3. 完整的训练循环（forward, loss, backward, step）
# 4. 能够运行并看到 loss 下降

import torch
import torch.nn as nn

# ↓ 你的代码在这里 ↓

# ↑ 你的代码在这里 ↑
```

### 任务 3: 一句话总结（10min）
用一句话向非技术人员解释：**"PyTorch 是做什么的？为什么 AI 工程师需要它？"**

---

## 📝 今日自检清单

- [ ] 我能写出带类型注解的函数签名
- [ ] 我能手写 Dataset 和 DataLoader
- [ ] 我能解释 `backward()` 和 `zero_grad()` 为什么必须配合使用
- [ ] 我能从零写出训练循环
- [ ] 我能解释 GPU 上 tensor 和 CPU 上 tensor 的区别
- [ ] 我知道 `.detach()` 的作用
- [ ] 我完成了费曼口头讲解

---

## 💡 遇到问题？

| 问题 | 解决路径 |
|------|----------|
| 广播报错 | 从最后一维对齐，检查维度是否兼容 |
| CUDA out of memory | 减小 batch_size，开启 gradient checkpointing |
| loss 不下降 | 检查学习率、数据归一化、是否忘记 zero_grad() |
| tensor 不在同一设备 | 统一 `.to(device)` |

---

> ✅ **完成打卡**: 全部 checklist 勾完后，在 README 中标记 Day 1 完成！
>
> 🔜 **明日预告**: 深度学习基础 — 理解神经元如何组成网络、CNN 如何"看"图像。
