# 🎯 Day 7: 行为克隆与模仿学习（8 小时）

> **口号**: "让机器人通过'看'来学习——模仿学习的理论与实践！"  
> **目标**: 理解 BC 原理和分布漂移问题，实现 ACT 的核心思想  
> **为什么重要**: 几乎所有的 VLA 训练都基于模仿学习范式

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | 行为克隆理论 | 1.5h | [ ] |
| 2 | 分布漂移与解决方案 | 1.5h | [ ] |
| 3 | Action Chunking 核心思想 | 1.5h | [ ] |
| 4 | 动手: 实现简单 BC | 1.5h | [ ] |
| 5 | ACT 论文速读 | 1h | [ ] |
| 6 | 费曼挑战 | 1h | [ ] |

---

## 模块 1: 行为克隆理论（1.5 小时）

### 1.1 BC 的数学本质 — 30min

```
行为克隆 (Behavior Cloning) = 监督学习

数据: D = {(o₁, a₁), (o₂, a₂), ..., (oₙ, aₙ)}
  o: 观测 (observation)
  a: 动作 (action) — 专家演示的动作

目标: 学一个策略 π(a|o)，使预测动作接近专家动作

训练: min E[(π(o) - a_expert)²]  ← MSE Loss

就是这么简单！但这有隐藏的问题...
```

### 1.2 从数据到模型 — 20min

```python
# BC 的最简单实现
class BehaviorCloning(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)

# 训练
def train_bc(model, dataloader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for obs, action in dataloader:
            pred_action = model(obs)
            loss = criterion(pred_action, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 这不就是 Day 2 的监督学习吗？是的！
```

### 1.3 BC 的优势和局限 — 20min

```
优势:
  ✅ 简单，就是监督学习
  ✅ 不需要 reward 函数（比 RL 简单）
  ✅ 数据效率高（充分利用每条演示）
  ✅ 训练稳定（没有 RL 的策略梯度问题）

局限:
  ❌ 分布漂移 (Distribution Shift) — 核心问题
  ❌ 不能超过专家水平
  ❌ 需要大量高质量演示数据
  ❌ 对多模态行为建模困难
```

### 1.4 为什么 BC 对于 VLA 是基础？ — 20min

```
RT-1, RT-2, OpenVLA 的训练本质上都是 BC:
  给定 (image, instruction) → 预测 expert action

但加上了:
  - 大规模数据（OXE）
  - 强大的视觉/语言 backbone
  - Action Tokenization（把连续动作变成离散 token）
  - 多任务混合训练
```

---

## 模块 2: 分布漂移与解决方案（1.5 小时）

### 2.1 分布漂移的直觉 — 30min

```
想象你正在学开车，教练带着你开:

训练数据: 教练开车的所有 (路况, 方向盘角度) 对
BC 模型: 学到的策略

问题:
  1. 模型出发时方向盘打偏了一点（小误差）
  2. 车驶入了训练数据中没有见过的路况区域
  3. 模型不知道怎么办 → 乱打方向
  4. 越来越偏 → 最终失控
  
这就是 Compounding Error（误差累积）！

数学表达:
  训练数据来自分布 p_data(o)
  但模型执行时访问的观测来自 p_π(o)
  p_data(o) ≠ p_π(o) ← 分布漂移
```

### 2.2 直观演示 — 20min

```python
# 模拟分布漂移的影响
import numpy as np
import matplotlib.pyplot as plt

def simulate_bc_failure(horizon=100):
    """
    假设专家走一条直线 y = 0
    BC 模型有微小误差
    """
    # 训练数据: 专家总是在 y=0 附近
    expert_data_x = np.random.uniform(0, 10, 1000)
    expert_data_y = np.zeros(1000) + np.random.normal(0, 0.01, 1000)

    # BC 模型: 学了但是有残差误差
    def bc_model(x, prev_y):
        return prev_y + np.random.normal(0, 0.05)  # 微小误差

    # 开环执行
    y = 0.0
    trajectory = [y]
    for t in range(horizon):
        y = bc_model(t/10, y)  # 每一步用上一步的预测结果
        trajectory.append(y)

    # 误差会指数级增长！
    print(f"初始误差: {trajectory[1]:.3f}")
    print(f"50步后误差: {trajectory[50]:.3f}")
    print(f"100步后误差: {trajectory[-1]:.3f}")

    return trajectory

# 运行这个，你会看到一条越来越发散到无穷的曲线
```

### 2.3 解决方案 — 30min

```
解决方案 1: DAGGER (Dataset Aggregation) — 2011 年
  1. 用 BC 训练初始策略 π
  2. 用 π 执行任务，记录 (o, π(o))
  3. 让专家标注这些 o 的正确动作 a*
  4. 把新数据 (o, a*) 加入训练集
  5. 重新训练 → 重复

  → 效果: p_π(o) 逐渐接近 p_data(o)
  → 局限: 需要专家在线标注，成本高

解决方案 2: Action Chunking — 2023 年
  一次预测未来的 K 个动作，而不是一个
  → 减少开环控制的频率
  → 每个 chunk 内部执行时不做新的决策
  → 降低累计误差的速度

解决方案 3: 数据增强
  在专家演示中添加噪声 → 让模型见过"偏离"的状态
  → 扩大训练分布

解决方案 4: 更好的架构
  - 用 Transformer 捕捉长程依赖
  - 用扩散模型表达多模态动作分布
```

---

## 模块 3: Action Chunking 核心思想（1.5 小时）

### 3.1 为什么需要 Chunking？ — 20min

```
传统 BC: 预测单个动作
  t=0: 看图像 → 预测 a₀ → 执行 a₀
  t=1: 看图像 → 预测 a₁ → 执行 a₁
  t=2: 看图像 → 预测 a₂ → 执行 a₂
  ...
  每一步都可能累积误差

Action Chunking: 预测未来的 K 个动作
  t=0: 看图像 → 预测 [a₀, a₁, ..., aₖ₋₁] → 执行前 m 个
  t=m: 看图像 → 预测 [aₘ, ..., aₘ₊ₖ₋₁] → 执行前 m 个
  ...
  减少了重新规划的频率，降低了误差累积

类似:
  走路时一直盯着脚下一步 → 容易走偏
  走路时看好前方 5 步的路径 → 走得更稳
```

### 3.2 Action Chunking 的数学 — 30min

```
传统 BC:
  π(a_t | o_t)

Action Chunking:
  π(a_{t:t+K} | o_t)  ← 一次预测 K 步

其中 K 是 prediction horizon（预测视野）

训练时:
  从专家轨迹中抽取连续的 K 个动作作为 target

推理时:
  预测 K 步，可以:
    1. 全部执行 K 步（开环）
    2. 只执行前 m 步（m < K），然后重新预测（有重叠）
```

```python
# Action Chunking 的数据准备
def prepare_chunked_data(trajectory, K=10):
    """
    trajectory: episode 包含 [obs_0, obs_1, ..., obs_T] 和 [act_0, act_1, ..., act_T]
    K: chunk 大小
    """
    chunks = []
    for t in range(len(trajectory) - K):
        obs = trajectory['obs'][t]
        actions = trajectory['actions'][t:t+K]  # 未来 K 个动作
        chunks.append((obs, actions))
    return chunks

# 模型输出
class ActionChunkingPolicy(nn.Module):
    def forward(self, obs):
        # 输入: 单帧观测
        # 输出: K 个动作
        return self.net(obs)  # [B, K * action_dim]
```

### 3.3 Temporal Ensemble — 20min

```
重叠执行时的集成技巧:

如果 K=5, m=1 (每次只执行 1 步):

时间 0: 预测 [a₀⁰, a₁⁰, a₂⁰, a₃⁰, a₄⁰] → 执行 a₀⁰
时间 1: 预测 [a₁¹, a₂¹, a₃¹, a₄¹, a₅¹] → 执行 a₁¹
时间 2: 预测 [a₂², a₃², a₄², a₅², a₆²] → 执行 a₂²

注意: a₁ 被预测了 2 次（时间 0 和 时间 1）

Temporal Ensemble: 用指数加权平均合并多次预测

weight(t) = exp(-λ * t)  ← 越晚预测的权重越小

最终 a₂ = Σ w_i * a₂^i / Σ w_i
```

---

## 模块 4: 动手实现简单 BC（1.5 小时）

### 完整小项目: BC 控制一个模拟环境

```python
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

# ===== 1. 收集专家数据（用随机数据模拟） =====
def collect_expert_data(env, episodes=10):
    """实际项目中这应该是人类遥操作的数据"""
    data = {"obs": [], "actions": []}

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # 使用环境内置的"专家"策略（或随机+规则）
            action = env.action_space.sample()  # 这里用随机代替
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            data["obs"].append(obs)
            data["actions"].append(action)
            obs = next_obs

    return data

# ===== 2. BC 模型 =====
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)

# ===== 3. 训练 =====
def train_bc(model, data, epochs=50, batch_size=64, lr=1e-3):
    obs = torch.FloatTensor(data["obs"])
    actions = torch.FloatTensor(data["actions"])

    dataset = torch.utils.data.TensorDataset(obs, actions)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_obs, batch_act in loader:
            pred = model(batch_obs)
            loss = criterion(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(loader))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {losses[-1]:.4f}")

    return losses

# ===== 4. 评估 =====
def evaluate_bc(model, env, episodes=5):
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = model(obs_tensor).squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    return total_reward / episodes

# ===== 5. 运行 =====
if __name__ == "__main__":
    env = gym.make("HalfCheetah-v4")
    print(f"Obs dim: {env.observation_space.shape[0]}")
    print(f"Action dim: {env.action_space.shape[0]}")

    data = collect_expert_data(env, episodes=20)
    print(f"Collected {len(data['obs'])} transitions")

    model = BCPolicy(
        env.observation_space.shape[0],
        env.action_space.shape[0]
    )
    losses = train_bc(model, data, epochs=50)

    avg_reward = evaluate_bc(model, env)
    print(f"Average reward: {avg_reward:.2f}")
```

**运行并回答**: Loss 下降了吗？策略能完成任务吗？（用随机数据的预期结果是什么？）

---

## 模块 5: ACT 论文速读（1 小时）

### 5.1 ACT 是什么？ — 20min

```
ACT (Action Chunking Transformer):
  论文: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
  机构: Stanford, 2023
  作者: Tony Zhao 等

核心创新:
  1. Action Chunking: 一次预测 K 个动作（K=100）
  2. CVAE 架构: 用条件 VAE 捕捉动作的多模态分布
  3. Transformer 编码: 时序动作序列用 Transformer 编码

为什么重要:
  - 在精细双手操作（穿针、拉链）上取得了惊人效果
  - 展示了大模型 + Action Chunking 的威力
  - 是 VLA 发展的重要里程碑
```

### 5.2 ACT 架构 — 25min

```
ACT 的核心架构（简化）:

        观测图像 o_t
            │
    ┌───────▼───────┐
    │  视觉编码器    │  (ResNet 或 ViT)
    │  + 关节状态    │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │ Transformer   │
    │  Encoder      │  ← 把观测编码成条件
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │ CVAE Decoder  │  ← 生成 K 步动作序列
    │ (Transformer) │
    └───────┬───────┘
            │
    动作序列 [a_t, ..., a_{t+K-1}]

CVAE (条件变分自编码器):
  训练时: 有编码器 q(z|a, o) 把动作编码成隐变量 z
         有解码器 p(a|z, o) 从隐变量重建动作
  推理时: 直接从先验 p(z|o) 采样 z，然后解码出动作
```

### 5.3 关键代码概念 — 15min

```python
# ACT 的核心损失 = 重建损失 + KL 散度
class ACTLoss(nn.Module):
    def forward(self, pred_actions, gt_actions, mu, logvar):
        # 重建损失
        recon_loss = F.l1_loss(pred_actions, gt_actions)

        # KL 散度: 让后验接近先验
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + beta * kl_loss  # β-VAE 风格
```

---

## 🎤 费曼挑战（1 小时）

### 任务 1: 白板讲解 (25min)
讲解以下内容（录音/录像）:
1. **行为克隆的工作原理**: 从数据到模型到部署，每一步干什么
2. **分布漂移**: 为什么微小的误差会导致灾难性失败？（用开车类比）
3. **Action Chunking 如何缓解分布漂移？**

### 任务 2: 代码审查 (20min)
审视你写的 BC 代码，回答:
1. 为什么用 MSE loss 而不是 Cross-Entropy？
2. 如果你要把这个 BC 改成 Action Chunking，需要改哪些地方？
3. 如果观测变成图像（不是状态向量），模型要怎么改？

### 任务 3: 论文一句话 (15min)
读完 ACT 论文的 abstract 和 introduction 后:
1. 用一句话总结 ACT 的核心贡献
2. ACT 和传统 BC 的三个关键区别
3. 为什么 ACT 用 Transformer 而不是 MLP？

---

## 📝 今日自检清单

- [ ] 我能解释行为克隆的数学公式
- [ ] 我能用代码实现一个 BC 模型
- [ ] 我理解分布漂移和 Compounding Error
- [ ] 我能说出 DAGGER 的工作原理
- [ ] 我理解 Action Chunking 与 Temporal Ensemble
- [ ] 我知道 ACT 的 CVAE 架构轮廓
- [ ] 我跑通了 BC 的简单实现
- [ ] 我完成了白板讲解

---

> ✅ **完成打卡**: BC + Action Chunking 已掌握！明天直击 VLA 核心！
>
> 🔜 **明日预告**: Day 8 — RT-1 和 RT-2 深度剖析，VLA 模型的完整理解！
