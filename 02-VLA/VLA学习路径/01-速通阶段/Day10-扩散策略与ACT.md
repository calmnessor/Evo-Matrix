# 🌊 Day 10: 扩散策略与 ACT 深入（8 小时）

> **口号**: "从噪声中'炼'出完美动作——理解扩散模型的魔法！"  
> **目标**: 理解扩散策略和 ACT 的原理，能够对比两种动作生成范式  
> **为什么重要**: 扩散策略是动作生成的重要范式，与自回归 VLA 互补

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | 扩散模型基础 | 1.5h | [ ] |
| 2 | Diffusion Policy 深入 | 1.5h | [ ] |
| 3 | ACT 论文深入 | 1.5h | [ ] |
| 4 | 自回归 vs 扩散对比 | 1h | [ ] |
| 5 | 动手：Diffusion Policy 简单实现 | 1.5h | [ ] |
| 6 | 费曼挑战 | 1h | [ ] |

---

## 模块 1: 扩散模型基础（1.5 小时）

### 1.1 扩散模型的直觉 — 30min

```
扩散模型 = "正向加噪 → 反向去噪"

想象做一杯咖啡:
  
  正向过程 (Forward Diffusion):
    清水 → 加一滴墨 → 加两滴墨 → ... → 全黑的水
    (信息逐步被破坏)
  
  反向过程 (Reverse Diffusion):
    全黑的水 → 去掉一点墨 → ... → 清水（咖啡）
    (从噪声中恢复出有用的结构)

在 VLA 中的类比:
  正向过程: 把一个完美的动作序列，逐步加入噪声，最终变成纯噪声
  反向过程: 从纯噪声开始，根据观测条件，逐步去噪，生成动作

为什么用于动作生成？
  → 可以表达复杂的多模态分布（一个观测可能对应多种合理动作）
  → 生成的动作天然平滑（去噪过程保证连续性）
  → 不需要像自回归那样逐 token 生成
```

### 1.2 DDPM 核心公式 — 30min

```
DDPM (Denoising Diffusion Probabilistic Models):

正向过程（不需要训练，手动定义）:
  x_t = √(α_t) * x_{t-1} + √(1-α_t) * ε,  ε ~ N(0, I)
  
  或者直接从 x_0 一步到位:
  x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε

反向过程（需要训练网络来预测）:
  训练: 给定 x_0 和 t，加噪得到 x_t
        让网络 ε_θ(x_t, t) 预测加入的噪声 ε
        Loss = MSE(ε_θ(x_t, t), ε)
  
  推理: 从 x_T ~ N(0, I) 开始
        逐步用网络去噪，直到 x_0

关键变量:
  t: 时间步 (1 到 T，通常 T=100 或 1000)
  α_t: 噪声调度参数
  ε: 高斯噪声
```

### 1.3 用代码理解 — 30min

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDiffusion(nn.Module):
    """
    最简单的扩散模型，用于生成 2D 点
    """
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T

        # 噪声调度: 线性从 beta_start 到 beta_end
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # 去噪网络: 简单的 MLP
        self.net = nn.Sequential(
            nn.Linear(2 + 1, 128),  # 2D 动作 + 1D 时间步
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),      # 预测噪声
        )

    def forward_diffusion(self, x_0, t):
        """正向加噪: x_0 → x_t"""
        # 从 0 到 T-1 取 batch_size 个随机时间步
        alpha_bar = self.alpha_bars[t]  # [B]
        alpha_bar = alpha_bar.view(-1, 1)

        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise

    def forward(self, x_0):
        """训练用: 预测噪声"""
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x_0.device)

        x_t, noise = self.forward_diffusion(x_0, t)

        # 将时间步编码加入输入
        t_emb = t.float().unsqueeze(-1) / self.T
        net_input = torch.cat([x_t, t_emb], dim=-1)

        pred_noise = self.net(net_input)

        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, batch_size=16):
        """推理用: 从噪声中采样"""
        x = torch.randn(batch_size, 2)  # 从纯噪声开始

        for t in reversed(range(self.T)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long)
            t_emb = t_batch.float().unsqueeze(-1) / self.T

            # 预测噪声
            net_input = torch.cat([x, t_emb], dim=-1)
            pred_noise = self.net(net_input)

            # DDPM 采样公式（简化）
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            if t > 0:
                z = torch.randn_like(x)
            else:
                z = 0

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * z

        return x

# 测试
model = SimpleDiffusion(T=100)
x_0 = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.0, 0.0]])  # 3个样本

loss = model(x_0)
print(f"Training loss: {loss:.4f}")

samples = model.sample(5)
print(f"Generated samples:\n{samples}")
```

---

## 模块 2: Diffusion Policy 深入（1.5 小时）

### 2.1 核心思想 — 30min

```
Diffusion Policy:
  论文: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
  机构: Columbia, Toyota Research Institute, MIT, 2023

核心公式:
  p(A_t | O_t) = 在给定观测 O_t 的条件下，从噪声中逐步去噪生成动作 A_t

A_t: 动作序列 [a_t, a_{t+1}, ..., a_{t+H-1}]  ← H 步动作
O_t: 观测（图像 + 机器人状态）

为什么不用自回归生成动作？
  自回归: p(a₁, a₂, ..., a_H) = p(a₁) * p(a₂|a₁) * ... * p(a_H|a₁:H-1)
  → 误差会累积，前面的小错导致后面大错

  扩散: 同时生成所有动作
  → 所有时间步的动作一起被去噪，保持全局一致性
  → 天然平滑（噪声逐步被去除）
```

### 2.2 架构对比 — 30min

```
Diffusion Policy 的两种架构变体:

1. CNN-based:
  [图像] → CNN编码器 → 特征
  [噪声动作 + 特征 + 时间步] → 1D CNN UNet → 去噪后动作
  
  优势: 简单，快
  劣势: 感受野有限

2. Transformer-based:
  [图像] → ViT编码器 → 特征
  [噪声动作 + 特征 + 时间步] → DiT (Diffusion Transformer) → 去噪后动作
  
  优势: 全局感受野，效果好
  劣势: 稍慢，需要更多数据
```

### 2.3 条件注入 — 20min

```python
class ConditionalDiffusionPolicy(nn.Module):
    """
    条件扩散策略: 在观测条件下生成动作
    """
    def __init__(self, obs_dim, action_dim, action_horizon=16):
        super().__init__()
        self.action_horizon = action_horizon

        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # 去噪网络: 输入 = [噪声动作, 观测特征, 时间步]
        self.denoiser = nn.Sequential(
            nn.Linear(action_horizon * action_dim + 256 + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_horizon * action_dim),
        )

    def forward(self, obs, actions_gt):
        """
        obs: [B, obs_dim]
        actions_gt: [B, action_horizon, action_dim]  真实动作序列
        """
        B = obs.shape[0]
        t = torch.randint(0, T, (B,), device=obs.device)

        # 1. 编码观测
        obs_feat = self.obs_encoder(obs)  # [B, 256]

        # 2. 对动作加噪
        actions_flat = actions_gt.view(B, -1)  # [B, H*D]
        x_t, noise = self.forward_diffusion(actions_flat, t)

        # 3. 去噪网络（条件 = 观测 + 时间）
        t_emb = t.float().unsqueeze(-1) / T
        net_input = torch.cat([x_t, obs_feat, t_emb], dim=-1)
        pred_noise = self.denoiser(net_input)

        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, obs):
        """给定观测，生成动作序列"""
        B = obs.shape[0]
        obs_feat = self.obs_encoder(obs)

        # 从噪声开始去噪
        x = torch.randn(B, self.action_horizon * action_dim)
        for t in reversed(range(T)):
            t_batch = torch.full((B,), t, dtype=torch.long)
            t_emb = t_batch.float().unsqueeze(-1) / T

            net_input = torch.cat([x, obs_feat, t_emb], dim=-1)
            pred_noise = self.denoiser(net_input)

            # DDPM 去噪一步
            x = denoise_step(x, pred_noise, t)

        return x.view(B, self.action_horizon, action_dim)
```

### 2.4 关键实验发现 — 10min

```
Diffusion Policy 论文的关键结果:

1. 动作序列的全局一致性 > 自回归
   - 扩散生成的动作序列更平滑
   - 没有自回归的误差累积

2. 对多模态分布的建模能力
   - 同一个观测可能对应"从左绕"和"从右绕"两种动作
   - 扩散模型能表达这种多模态，自回归会取平均

3. 推理速度
   - 需要多步去噪（通常 10-100 步）
   - 比单次 forward 慢，但比自回归逐 token 生成快
```

---

## 模块 3: ACT 论文深入（1.5 小时）

### 3.1 ACT 回顾与深入 — 30min

```
ACT (Action Chunking Transformer) 的完整理解:

与 Diffusion Policy 的相似之处:
  - 都预测动作序列（chunk）
  - 都用了 Transformer

本质区别:

  ACT = CVAE (条件变分自编码器)
    编码器 q(z|action_chunk, observation):
      看到真实动作后，编码成隐变量 z
    解码器 p(action_chunk|z, observation):
      从隐变量 z 和观测，重建动作
    推理时:
      从先验 p(z|observation) 采样 z → 解码动作
  
  Diffusion Policy = 条件扩散模型
    正向: 加噪
    反向: 去噪（条件是观测）
    → 不需要从先验采样 z
    → 通过多步去噪逐步细化动作
```

### 3.2 CVAE 训练详解 — 30min

```python
class ACTCVAE(nn.Module):
    """
    ACT 的 CVAE 核心（简化版）
    """
    def __init__(self, obs_dim, action_dim, K=100, latent_dim=32):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim

        # 观测编码器
        self.obs_encoder = TransformerEncoder(obs_dim, d_model=512)

        # 编码器: q(z | action, observation)
        self.encoder = nn.Sequential(
            nn.Linear(K * action_dim + 512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2),  # mu + logvar
        )

        # 解码器: p(action | z, observation)
        self.decoder = TransformerDecoder(
            d_model=512 + latent_dim,
            output_dim=K * action_dim,
        )

    def forward(self, obs, actions_gt):
        """
        obs: [B, obs_dim]
        actions_gt: [B, K, action_dim]
        """
        B = obs.shape[0]
        K = self.K

        # 1. 编码观测
        obs_feat = self.obs_encoder(obs)  # [B, 512]

        # 2. 编码器: 用真实动作来编码 z
        actions_flat = actions_gt.view(B, -1)  # [B, K*D]
        enc_input = torch.cat([actions_flat, obs_feat], dim=-1)
        enc_out = self.encoder(enc_input)

        mu = enc_out[:, :self.latent_dim]
        logvar = enc_out[:, self.latent_dim:]

        # 重参数化:
        # z = mu + sigma * epsilon, epsilon ~ N(0, I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # 3. 解码器: 从 z 重建动作
        dec_input = torch.cat([obs_feat, z], dim=-1)
        pred_actions = self.decoder(dec_input)  # [B, K*D]

        # 4. 损失
        recon_loss = F.l1_loss(pred_actions, actions_flat)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

        # β-VAE 风格: kl_weight 控制 KL 的权重
        kl_weight = 10.0
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

    @torch.no_grad()
    def infer(self, obs):
        """推理: 从先验采样 z，解码动作"""
        B = obs.shape[0]
        obs_feat = self.obs_encoder(obs)

        # 从标准正态先验采样 z
        z = torch.randn(B, self.latent_dim).to(obs.device)

        dec_input = torch.cat([obs_feat, z], dim=-1)
        pred_actions = self.decoder(dec_input)

        return pred_actions.view(B, self.K, -1)
```

### 3.3 ACT 的关键实验 — 15min

```
ACT 展示的惊人能力（Stanford 实际机器人实验）:

任务:
  1. 穿鞋带（需要双手协调）
  2. 拉拉链
  3. 套挂环

关键发现:
  - K=100 的 chunk 显著优于 K=1
  - CVAE > 确定性回归（因为动作的多模态性）
  - 数据收集成本低（用 $100 的手持夹爪示教器）
  - 只用 50 条演示就能学得很好
```

### 3.4 ACT 的开源项目结构 — 15min

```
act/
├── imitate_episodes.py     ← 主训练脚本
├── policy.py               ← ACT 策略定义
├── detr/                   ← DETR Transformer
├── constants.py            ← 配置常量
├── utils.py                ← 工具函数
└── visualize_episodes.py   ← 可视化

关键配置:
  - chunk_size: 100 (K)
  - lr: 1e-5
  - kl_weight: 10
  - temporal_agg: True (时序集成)
```

---

## 模块 4: 自回归 vs 扩散 — 动作生成范式对比（1 小时）

### 4.1 技术对比 — 30min

```
┌───────────────┬──────────────────┬──────────────────┐
│   维度         │  自回归 (RT-2等)  │  扩散 (Diffusion Policy等) │
├───────────────┼──────────────────┼──────────────────┤
│ 生成方式       │ 逐 token/逐步生成  │ 从噪声全局去噪     │
│ 推理速度       │ N 步 forward      │ M 步 forward      │
│               │ (N=动作维度)      │ (M=10-100)        │
│ 动作平滑性     │ 需要额外约束      │ 天然平滑          │
│ 多模态能力     │ 弱（趋向取平均）   │ 强（天然多模态）   │
│ 条件注入       │ 拼接/交叉注意力   │ 拼接/交叉注意力    │
│ 可扩展性       │ 可接 LLM 推理    │ 单独训练          │
│ 模型大小       │ 大（7B+）         │ 小（100M-1B）     │
│ 代表模型       │ RT-2, OpenVLA    │ Octo, DP          │
└───────────────┴──────────────────┴──────────────────┘
```

### 4.2 实际选择建议 — 20min

```
什么时候用自回归 VLA (RT-2/OpenVLA):
  ✅ 需要语言推理和常识（"拿起危险的那个"）
  ✅ 任务涉及多步规划
  ✅ 有足够的计算资源（大 GPU）
  ✅ 数据包含丰富的语言指令

什么时候用扩散策略:
  ✅ 需要精细的连续动作控制
  ✅ 任务是多模态的（同一个场景有多个合理动作）
  ✅ 计算资源有限
  ✅ 以视觉为主的策略，语言不重要

趋势: 融合！
  π₀ 已经结合了 VLM + Flow Matching（连续时间的扩散）
  → 未来可能是 "大 VLM 理解 + 扩散生成动作"
```

---

## 模块 5: 动手 — 简单 Diffusion Policy 实现（1.5 小时）

### 完整实现:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ===== 完整可运行的 Diffusion Policy =====

class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim=2, action_dim=2, action_horizon=16,
                 T=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()

        self.T = T
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # 噪声调度
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, T))
        alphas = 1 - self.betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', torch.cumprod(alphas, dim=0))

        # 观测编码器
        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        # 去噪网络 (UNet风格的MLP)
        input_dim = action_horizon * action_dim + 256  # 噪声动作 + 观测特征
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_horizon * action_dim),
        )

        # 时间步编码
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def forward(self, obs, actions_gt):
        """
        obs: [B, obs_dim]
        actions_gt: [B, action_horizon, action_dim]
        """
        B = obs.shape[0]
        device = obs.device

        # 随机时间步
        t = torch.randint(0, self.T, (B,), device=device)

        # 编码观测
        obs_feat = self.obs_net(obs)  # [B, 256]

        # 对动作加噪
        actions_flat = actions_gt.view(B, -1)
        alpha_bar = self.alpha_bars[t].view(-1, 1)

        noise = torch.randn_like(actions_flat)
        x_t = torch.sqrt(alpha_bar) * actions_flat + torch.sqrt(1 - alpha_bar) * noise

        # 预测噪声
        net_input = torch.cat([x_t, obs_feat], dim=-1)
        pred_noise = self.denoiser(net_input)

        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, obs):
        """从观测生成动作序列"""
        B = obs.shape[0]
        device = obs.device

        obs_feat = self.obs_net(obs)

        # 从噪声开始
        x = torch.randn(B, self.action_horizon * self.action_dim, device=device)

        # 逐步去噪
        for t in reversed(range(self.T)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)

            net_input = torch.cat([x, obs_feat], dim=-1)
            pred_noise = self.denoiser(net_input)

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            if t > 0:
                z = torch.randn_like(x)
            else:
                z = 0

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * z

        return x.view(B, self.action_horizon, self.action_dim)


# ===== 简单测试: 生成 2D 轨迹 =====

def generate_synthetic_data(n_samples=1000):
    """生成合成数据: 从不同起点到原点的轨迹"""
    obs_list = []
    action_list = []

    for _ in range(n_samples):
        # 随机起点
        start = np.random.uniform(-2, 2, 2)
        # 目标: 到达原点
        target = np.array([0.0, 0.0])

        # 生成一条直线轨迹
        trajectory = []
        for t in range(16):
            alpha = t / 15.0
            point = start * (1 - alpha) + target * alpha
            point += np.random.normal(0, 0.02, 2)  # 小噪声
            trajectory.append(point)

        obs_list.append(start)
        action_list.append(np.array(trajectory))

    return (
        torch.FloatTensor(np.array(obs_list)),
        torch.FloatTensor(np.array(action_list)),
    )

# 训练
obs_data, action_data = generate_synthetic_data(1000)
dataset = torch.utils.data.TensorDataset(obs_data, action_data)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

model = DiffusionPolicy(obs_dim=2, action_dim=2, action_horizon=16, T=50)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    for obs_batch, act_batch in loader:
        loss = model(obs_batch, act_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 测试生成
test_obs = torch.FloatTensor([[1.5, 1.5]])
generated = model.sample(test_obs).squeeze(0).numpy()

plt.figure(figsize=(6, 6))
plt.plot(generated[:, 0], generated[:, 1], 'b-o', markersize=3)
plt.scatter(0, 0, c='r', s=100, marker='*', label='Target')
plt.scatter(1.5, 1.5, c='g', s=100, marker='o', label='Start')
plt.legend()
plt.title("Generated Trajectory")
plt.grid(True)
plt.axis('equal')
plt.savefig('diffusion_policy_output.png')
print("Trajectory saved to diffusion_policy_output.png")
```

---

## 🎤 费曼挑战（1 小时）

### 任务 1: 直觉讲解 (20min)
用最朴素的语言解释:
- "扩散模型就像一个雕塑家，从一个石块（噪声）开始，逐步凿出雕像（动作）"
- 用这个类比讲解正向过程和反向过程

### 任务 2: 公式默写 (20min)
在纸上默写 DDPM 的:
1. 正向加噪公式
2. 训练损失公式
3. 采样公式
每个公式旁边写一句话解释它的作用

### 任务 3: 范式对比 (20min)
口述 3 分钟:
"如果现在让你选一个方案做机器人的 VLA 策略，你会选自回归还是扩散？为什么？"
要求至少列出 3 个理由。

---

## 📝 今日自检清单

- [ ] 我理解扩散模型的"正向加噪 → 反向去噪"
- [ ] 我能写出 DDPM 的训练损失公式
- [ ] 我理解 Diffusion Policy 相对于自回归的优势
- [ ] 我理解 ACT 的 CVAE 训练方式
- [ ] 我知道 KL 散度在 CVAE 中的作用
- [ ] 我跑通了简单 Diffusion Policy 实现
- [ ] 我能对比自回归和扩散在动作生成中的优劣
- [ ] 我完成了公式默写

---

> ✅ **完成打卡**: 扩散策略已掌握！明天实战项目！
>
> 🔜 **明日预告**: Day 11 — 实战项目 1，OpenVLA 部署与微调！
