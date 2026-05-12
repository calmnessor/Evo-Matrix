# PyTorch 实战：完整训练循环

> 从零搭建一个 VLA 风格策略网络的训练循环。

---

## 完整训练循环

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. 定义一个简单的 VLA 策略网络 (简化版)
class SimplePolicy(nn.Module):
    def __init__(self, obs_dim=128, act_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, act_dim),  # 输出: 7-DOF 动作
        )
    
    def forward(self, x):
        return self.net(x)

# 2. 准备数据 (模拟: 1000 条 观测→动作 的演示数据)
X = torch.randn(1000, 128)  # 1000 条观测
Y = torch.randn(1000, 7)    # 对应的 7-DOF 动作
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. 初始化
model = SimplePolicy()
criterion = nn.MSELoss()                      # 连续动作回归
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 4. 训练循环
for epoch in range(100):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        
        # 梯度裁剪 (防止爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(loader):.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

# 5. 推理
model.eval()
with torch.no_grad():
    test_obs = torch.randn(1, 128)
    action = model(test_obs)
    print(f"\n测试: 观测 → 动作 = {action.squeeze().numpy()}")
```

---

## 训练循环模板 (可复用)

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()
    return total_loss / len(loader)
```

---

## 常见问题排查

| 症状 | 可能原因 | 排查方法 |
|------|---------|---------|
| Loss 不下降 | lr 太小或太大 | 尝试 lr ∈ {1e-3, 1e-4, 1e-5} |
| Loss NaN | 梯度爆炸 | 加 gradient clipping，降低 lr |
| 训练 loss 低但验证 loss 高 | 过拟合 | 加 Dropout/Weight Decay/数据增强 |
| Loss 震荡剧烈 | batch size 太小或 lr 太大 | 增大 batch 或降低 lr |
| 显存不够 | batch size 太大 | 减小 batch + gradient accumulation |
| 训练忽快忽慢 | 数据加载成瓶颈 | 增大 num_workers，检查 I/O |

---

## 关联笔记

- [[深度学习基础]] — 反向传播、优化器原理
- [[训练技巧与正则化]] — 归一化、学习率调度、正则化方法
- [[../大语言模型与微调/QLoRA完整实现|QLoRA完整实现]] — VLA 微调的完整训练脚本
