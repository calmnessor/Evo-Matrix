# 🧠 Day 2: 深度学习基础速通（8 小时）

> **口号**: "理解神经元，才能指挥千军万马的参数！"  
> **目标**: 理解 MLP/CNN/反向传播，能独立训练一个 CNN  
> **核心心法**: 深度学习 = 函数拟合 + 梯度下降 + 大规模数据

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | MLP 与激活函数 | 1h | [ ] |
| 2 | 损失函数全家桶 | 0.5h | [ ] |
| 3 | 反向传播彻底理解 | 1.5h | [ ] |
| 4 | 优化器与学习率调度 | 1h | [ ] |
| 5 | CNN 卷积神经网络 | 2h | [ ] |
| 6 | 动手实战：训练 CNN | 1h | [ ] |
| 7 | 费曼挑战 | 1h | [ ] |

---

## 模块 1: MLP 与激活函数（1 小时）

### 1.1 神经元 = 带激活函数的线性变换 — 20min

```
输入 x → [线性变换: Wx + b] → [激活函数 σ] → 输出 y

单层: y = σ(Wx + b)
多层: y = σ₂(W₂ · σ₁(W₁x + b₁) + b₂)
```

```python
# 一个 MLP 的本质
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
```

**关键问题**: 如果所有激活函数都是线性的（σ(x)=x），多层 MLP 等价于什么？

### 1.2 激活函数对比 — 20min

| 函数 | 公式 | 特点 | 使用场景 |
|------|------|------|----------|
| **ReLU** | max(0, x) | 简单、梯度不饱和 | CNN、MLP |
| **GELU** | x·Φ(x) | 平滑、性能好 | **Transformer/VLA** |
| **SiLU** | x·σ(x) | 又叫 Swish | Llama 系列 |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 输出 (-1,1) | LSTM、动作输出层 |
| **Softmax** | eˣⁱ/Σeˣ | 输出概率分布 | 分类、注意力权重 |

```python
# VLA 中的实际用法
# 视觉 Transformer 用 GELU
vision_tower:
  hidden_act: "gelu"

# 动作输出用 Tanh（归一化到 [-1, 1]）
action = nn.Tanh()(action_head(x))  # 机械臂关节角度范围
```

### 1.3 为什么深度重要？ — 20min

> 单层 MLP 只能拟合线性函数 → 深度网络可以拟合任意复杂函数（Universal Approximation Theorem）

```
浅层 (1-2 层): 只能学简单的特征组合
中层 (3-10 层): 可以学层次化特征  (边缘 → 形状 → 物体)
深层 (10-100+ 层): 可以学高度抽象的语义概念
```

---

## 模块 2: 损失函数全家桶（0.5 小时）

### 2.1 VLA 中最常用的损失函数

```python
# 1. MSE (Mean Squared Error) — 动作预测最常用
loss = nn.MSELoss()(pred_action, gt_action)
# 含义: 预测的动作值与真实值越接近，loss 越小

# 2. L1 Loss — 对离群点不那么敏感
loss = nn.L1Loss()(pred_action, gt_action)

# 3. Cross-Entropy — 用于离散动作（如抓取/不抓取）
loss = nn.CrossEntropyLoss()(logits, discrete_action_label)

# 4. BCE (Binary Cross Entropy) — 用于二分类（如是否接触物体）
loss = nn.BCEWithLogitsLoss()(contact_logits, contact_label)

# 5. Smooth L1 — 结合 MSE 和 L1 的优点
loss = nn.SmoothL1Loss()(pred_action, gt_action)  # ACT 论文中用到
```

**自检**: VLA 为什么一般用 MSE 而不是 Cross-Entropy？

---

## 模块 3: 反向传播彻底理解（1.5 小时）

> ⭐ **这是深度学习最重要的 1.5 小时！**

### 3.1 链式法则 = 反向传播的本质 — 30min

```
前向传播:  x ──→ a=f(w₁x) ──→ b=g(w₂a) ──→ L=loss(b)
反向传播:  dL/dw₁ = dL/db · db/da · da/dw₁
            ↑ 最终梯度 = 上层的梯度 × 本层的局部导数

链式法则: 每个参数 w 的梯度 = loss 对 w 的偏导数
          沿着计算图从输出往输入方向逐层传播
```

**画图理解**（请一定手画一遍）:
```
    x ──[×w₁]──→ h₁ ──[ReLU]──→ a ──[×w₂]──→ h₂ ──[MSE]──→ L
    │              │              │              │              │
    └── dL/dw₁? ───┘              └── dL/dw₂? ───┘              │
        = dL/da · da/dh₁ · dh₁/dw₁   = dL/dh₂ · dh₂/dw₂
```

### 3.2 手算一个简单例子 — 30min

```
给定: x=2, w₁=3, w₂=1, 真值 y=10

前向:
  h = w₁ · x = 6
  a = ReLU(6) = 6
  ŷ = w₂ · a = 6
  L = (ŷ - y)² = (6-10)² = 16

反向:
  dL/dŷ = 2(6-10) = -8
  dL/dw₂ = dL/dŷ · dŷ/dw₂ = -8 · 6 = -48
  dL/da = dL/dŷ · dŷ/da = -8 · 1 = -8
  dL/dw₁ = dL/da · da/dh · dh/dw₁ = -8 · 1 · 2 = -16

更新 (lr=0.1):
  w₁ = 3 - 0.1·(-16) = 4.6
  w₂ = 1 - 0.1·(-48) = 5.8
```

**自检**: 手算完再验算一遍，确保每个数字都理解从哪里来的。

### 3.3 从数学到代码 — 30min

```python
# PyTorch 自动反向传播等价于什么？

# 手动实现反向传播（只看不做，理解即可）
class ManualLinear:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None  # 缓存前向的输入

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad_output):
        # dL/dw = x^T @ grad_output
        self.grad_w = self.x.T @ grad_output
        # dL/dx = grad_output @ w^T （传给上一层）
        self.grad_x = grad_output @ self.w.T
        return self.grad_x
```

**关键洞察**: PyTorch 的 Autograd 就是在你调用 `loss.backward()` 时自动做了这一切，并且处理了各种复杂情况（共享参数、in-place 操作等）。

---

## 模块 4: 优化器与学习率调度（1 小时）

### 4.1 从 SGD 到 AdamW — 30min

```
SGD:       w = w - lr · ∇L        最简单，但收敛慢
Momentum:  v = β·v + ∇L          加惯性，越过局部最优
           w = w - lr · v
Adam:      m = β₁·m + (1-β₁)·∇L   一阶 + 二阶动量
           v = β₂·v + (1-β₂)·∇L²  自适应学习率
           w = w - lr · m/(√v + ε)
AdamW:     同 Adam，但权重衰减单独处理（VLA 标配）
```

```python
# VLA 训练标准配置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,           # 初始学习率
    weight_decay=0.01, # 权重衰减（L2 正则化）
    betas=(0.9, 0.999),
)

# 配合学习率预热 + 余弦衰减
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,   # 前 500 步从 0 线性增长到 lr
    num_training_steps=10000,
)
```

### 4.2 学习率调度策略 — 20min

```python
# 三种常见策略
# 1. 余弦退火 (Cosine Annealing) — 最常用
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 2. 阶梯衰减 (Step LR) — 简单粗暴
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 3. Warmup + Cosine — 大模型标配（先热身再衰减）
# 见上面 transformers 的例子
```

### 4.3 梯度裁剪 — 10min

```python
# 防止梯度爆炸，VLA 训练必加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 模块 5: CNN 卷积神经网络（2 小时）

> VLA 中虽然 ViT 逐渐取代 CNN，但理解 CNN 是理解视觉特征提取的基础

### 5.1 卷积的本质 — 30min

```
卷积 = 用小窗口在图像上滑动，每个位置算加权和

输入 [H × W × C]
  ↓ 卷积核 [k × k × C × C'] 滑动
输出 [H' × W' × C']

关键参数:
- kernel_size: 窗口大小（3×3 最常见）
- stride: 步长（2 用于下采样）
- padding: 填充（保持尺寸）
- dilation: 空洞卷积（增大感受野）
```

```python
# 一个典型的 CNN 特征提取器
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 224 → 112
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),  # 112 → 56

            # ResNet 风格的残差块
            ResidualBlock(64, 128, stride=2),      # 56 → 28
            ResidualBlock(128, 256, stride=2),     # 28 → 14
            ResidualBlock(256, 512, stride=2),     # 14 → 7

            nn.AdaptiveAvgPool2d(1),               # 7 → 1
            nn.Flatten(),
        )

    def forward(self, x):  # [B, 3, 224, 224]
        return self.conv(x)  # [B, 512]
```

### 5.2 残差连接 — 20min

这是 Transformer 中残差连接的来源，必须理解。

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_c, out_c, 1, stride) if stride != 1 else nn.Identity()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)  # 🔑 残差连接: F(x) + x
        return F.relu(out)

# 为什么有效？梯度可以通过 shortcut 直接传到前面
# d(F(x)+x)/dx = dF(x)/dx + 1 ← 这个 "+1" 防止梯度消失
```

### 5.3 BatchNorm 与 LayerNorm — 20min

```
BatchNorm:  对 [B, C, H, W] 的 B,H,W 维度归一化
            适合 CNN，不适合 Transformer/VLA（batch 太小会不稳定）

LayerNorm:  对 [B, L, D] 的 D 维度归一化
            适合 Transformer/VLA（不依赖 batch 统计）
```

```python
# VLA 中基本都是 LayerNorm，几乎不用 BatchNorm
self.norm = nn.LayerNorm(hidden_dim)  # VLA 标配
```

### 5.4 感受野 (Receptive Field) — 10min

```
感受野 = 输出特征图上的一个点对应输入图像上多大的区域

浅层 CNN: 感受野小（看到的是边缘、纹理）
深层 CNN: 感受野大（看到的是物体、场景）

VLA 需要大感受野理解场景 → 这也是为什么 VLA 转向 ViT
（ViT 的 self-attention 天然有全局感受野）
```

---

## 模块 6: 动手实战 — 训练 CNN（1 小时）

### 任务: 训练一个 CNN 做图像分类（CIFAR-10）

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 1. 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 2. 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 3. 训练（用 Day 1 学到的训练循环！）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

**目标**: 5 个 epoch 后 loss 应该降到 0.5 以下。

---

## 🎤 费曼挑战（1 小时）

### 任务 1: 白板讲解（30min）
找一块白板/纸，画出以下内容并口头讲解:

1. **反向传播的全过程**: 画一个 2 层网络，标注前向和反向的每一步数值
2. **卷积操作**: 画一个 5×5 输入、3×3 卷积核、步长 1 的例子
3. **残差连接为什么有效**: 用梯度流动的角度解释

### 任务 2: 类比挑战（15min）
用生活中的事物类比以下概念:
- 学习率 → ?（提示：步长）
- 梯度 → ?（提示：坡度）
- 过拟合 → ?（提示：死记硬背）
- Dropout → ?（提示：随机请假）

### 任务 3: 一句话总结（15min）
回答一个经典面试题：**"请向一个高中生解释深度学习是如何工作的"**

---

## 📝 今日自检清单

- [ ] 我能手写一个 MLP 的 forward
- [ ] 我能手算一个 2 层网络的反向传播
- [ ] 我知道 AdamW 比 SGD 好在哪里
- [ ] 我能解释卷积、padding、stride
- [ ] 我能解释残差连接和梯度消失的关系
- [ ] 我知道 BatchNorm 和 LayerNorm 的区别和使用场景
- [ ] 我成功训练了 CIFAR-10 CNN 并看到 loss 下降
- [ ] 我完成了费曼白板讲解

---

> [!info] 知识库关联
> - [[../../../01-基础理论/深度学习基础/深度学习基础|深度学习基础]] — 核心概念与数学推导
> - [[../../../01-基础理论/深度学习基础/PyTorch实战|PyTorch实战]] — 完整训练循环代码
> - [[../../../01-基础理论/深度学习基础/训练技巧与正则化|训练技巧与正则化]] — Normalization/激活/防过拟合
> - [[../../../01-基础理论/深度学习基础/表示学习与Embedding|表示学习与Embedding]] — 从像素到语义向量
>
> ✅ **完成打卡**: 完成后在 README 中勾选 Day 2！
>
> 🔜 **明日预告**: Transformer 架构 — 理解 Attention 机制，手写 Mini Transformer！
