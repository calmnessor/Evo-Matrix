# ACT (Action Chunking Transformer)

> 动作分块 + 条件 VAE → 2023 年机器人学习的标志性方法。一次预测多个未来动作，用 VAE 的随机性覆盖动作的多模态。

## 一句话总结

**ACT = CVAE + Action Chunking + Transformer Encoder-Decoder**

把动作生成变为"编码→采样→解码"的过程：从观测编码出隐变量 z，再从 z 解码出未来多步动作。

## 为什么需要 ACT？

### 普通 BC 的三个痛点

1. **分布漂移**：一步一步预测 → 每步误差累积 → 逐渐偏离训练分布 → 崩溃
2. **动作多模态**：同一观测下可以有多种正确动作（左绕和右绕都可以）→ MSE loss 学出"平均轨迹"= 撞中间
3. **高频推理负担**：50Hz 控制需要每秒推理 50 次 → Transformer 做不到

### ACT 的三招应对

| 痛点 | ACT 方案 | 原理 |
|------|---------|------|
| 分布漂移 | **Action Chunking** | 一次预测 16 步，取前 8 步执行 → 减少推理与执行的频率差 |
| 动作多模态 | **CVAE** | latent z 包含动作的多种可能 → 采样不同的 z 得到不同动作 |
| 推理负担 | **Chunking + Overlap** | 8 步推理一次 → 50Hz 执行 |


## 核心三件套深度拆解

### 1. CVAE (Conditional Variational Autoencoder)

**普通 AE vs VAE vs CVAE**：

```
AE:      x → Encoder → z → Decoder → x̂
         目标: 重建 x，z 是确定性瓶颈

VAE:     x → Encoder → (μ, σ) → 采样 z ~ N(μ, σ) → Decoder → x̂
         目标: 重建 x + KL(N(μ,σ) || N(0,I))，z 是随机变量

CVAE:    (x, c) → Encoder → (μ, σ) → 采样 z → Decoder(c, z) → x̂
         目标: 在条件 c 下重建 x，z 编码"除 c 以外的变化因素"
```

在 ACT 中：
- $x$ = 真实动作序列（训练时才有）
- $c$ = 观测（图像 + 关节角）
- $z$ = 动作风格的隐变量（左绕 vs 右绕、快 vs 慢）

**KL Loss 的作用**：
$$\mathcal{L}_{KL} = D_{KL}(q(z|x,c) \| p(z|c)) = D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I))$$

- 约束后验 $q(z|x,c)$ 接近先验 $p(z|c) = \mathcal{N}(0, I)$
- 推理时没有真实动作 → 直接从 $\mathcal{N}(0, I)$ 采样 z
- KL 确保"训练时从后验采"和"推理时从先验采"分布一致

**β-VAE 的 β**：ACT 总 loss = MSE(重建) + β × KL。β 控制"重建精度 vs 先验匹配"的权衡。β 太大 → 模型忽略 z，退化为普通 BC。

### 2. Action Chunking

```
时间线 →

预测窗口 (16 步):
  ████████████████████  t 时刻一次预测 t 到 t+15 的所有动作

执行窗口 (8 步):
  ████████████          取前 8 步执行到 t+7

滑动窗口:
  t:    ████████████████████  预测 [t, t+15]
        执行 ████████████     执行 [t, t+7]
  t+8:       ████████████████████  预测 [t+8, t+23]
             执行 ████████████     执行 [t+8, t+15]
  重叠的 [t+8, t+15] 取两次预测的平均 → Temporal Ensemble → 更平滑
```

**为什么有效？**
- 执行频率 >> 推理频率（8:1 倍加速）
- 重叠取平均 = 天然的平滑滤波
- 一次预测长序列 → 模型必须学到连贯的动作模式

### 3. Transformer Encoder-Decoder

```
Encoder:
  输入: 当前观测 (图像 + 关节角 + 可选语言)
  处理: Self-Attention → 上下文感知的观测特征
  输出: obs_feat [1, d_model]

StyleDrop (条件注入):
  z → MLP → (style_vector)
  Decoder 的每层 Cross-Attention 前注入 z 的信息
  类似 FiLM，但通过 Decoder 的 Cross-Attention 实现

Decoder:
  输入: 可学习的 Position Embedding [T, d_model]
  Self-Attention: 动作序列内部的时序依赖
  Cross-Attention: Q←Decoder, K/V←Encoder → 每个动作步"看"观测
  输出: 动作序列 [T, action_dim]
```

## 架构完整流程

```
训练阶段:
  图像 → ResNet → Encoder → obs_feat ─────────────┐
                                                     ├→ Decoder → 重建动作
  真实动作序列 → Action Encoder → (μ, σ) → z ──────┘
                                                     
  Loss = ||action - reconstructed||² + β × KL(N(μ,σ) || N(0,I))
                   ↑ MSE                       ↑ 正则化

推理阶段:
  图像 → ResNet → Encoder → obs_feat ─┐
                                       ├→ Decoder → 动作序列 [T, act_dim]
  从 N(0, I) 采样 → z ───────────────┘
  
  取前 K 步执行，滑动窗口重新预测 + 重叠平均
```

## 代码骨架

```python
class ACTPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_size=16, latent_dim=32):
        super().__init__()
        self.chunk_size = chunk_size
        
        # Encoder: 观测 → 特征
        self.obs_encoder = TransformerEncoder(obs_dim, d_model=256)
        
        # Action Encoder (训练时): 真实动作 → (μ, σ)
        self.action_encoder = nn.Sequential(
            nn.Linear(act_dim * chunk_size, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: obs_feat + z → 动作序列
        self.decoder = TransformerDecoder(
            d_model=256,
            chunk_size=chunk_size,
            act_dim=act_dim
        )
    
    def encode(self, obs, actions=None):
        obs_feat = self.obs_encoder(obs)
        
        if actions is not None:  # 训练模式
            h = self.action_encoder(actions.flatten(1))
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
            return obs_feat, z, mu, logvar
        else:  # 推理模式
            z = torch.randn(obs.size(0), self.latent_dim).to(obs.device)
            return obs_feat, z, None, None
    
    def forward(self, obs, actions=None):
        obs_feat, z, mu, logvar = self.encode(obs, actions)
        action_pred = self.decoder(obs_feat, z)  # [B, chunk_size, act_dim]
        return action_pred, mu, logvar


# 训练
def train_step(model, obs, actions, beta=0.01):
    pred_actions, mu, logvar = model(obs, actions)
    
    mse_loss = F.mse_loss(pred_actions, actions)
    
    # KL Loss (闭合解，标准 VAE 公式)
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    
    total_loss = mse_loss + beta * kl_loss
    return total_loss
```

## ACT vs Diffusion Policy

| 维度 | ACT | Diffusion Policy |
|------|-----|-----------------|
| 生成方式 | VAE Decoder 一步生成 | 扩散去噪 N 步 (通常 10-100) |
| 推理速度 | **快** (1 step) | 慢 (N steps, DDIM 可加速到 ~10) |
| 动作多峰 | 有限（z 采样的随机性） | **强**（扩散天然建模多模态分布） |
| 动作平滑 | 好 (Chunking + overlapping) | **很好** (扩散天然平滑) |
| 训练复杂度 | 中 (VAE + KL 调参) | 中 (去噪预测) |
| 适合场景 | 推理延迟敏感，动作多样性要求不高 | 精细操作，需要多模态动作 |

## 训练技巧

| 问题 | 可能原因 | 解决 |
|------|---------|------|
| KL loss 快速趋零 | β 太大 → z 被忽略 | 减小 β (1e-3)，或用 KL annealing |
| 重建 loss 不降 | Encoder 容量不足 | 加大 d_model，或换更大的视觉 backbone |
| 推理时动作抖动 | z 采样的方差太大 | 推理时用 μ (不用 σ 采样)，或用更小的 latent_dim |
| 动作太平滑/保守 | overlapping 太多 | 减小执行窗口/预测窗口的比率 |
| 分布漂移仍然存在 | chunk size 不够 | 增大 chunk_size，或结合 DAGGER |

## 推荐阅读

| # | 资源 | 关键词 |
|---|------|--------|
| 1 | **[ACT 原论文](https://arxiv.org/abs/2304.13705)** | Action Chunking, CVAE, Temporal Ensemble |
| 2 | **[ACT 官方代码](https://github.com/tonyzhaozh/act)** | PyTorch 实现，含训练脚本 |
| 3 | **[ALOHA 项目页](https://tonyzhaozh.github.io/aloha/)** | ACT 的低成本遥操作硬件 |

## 自检问题

### 基础关
- [ ] 我理解 CVAE 和普通 AE 的核心区别（随机 latent + KL 正则）
- [ ] 我能解释 KL Loss 在 VAE 中的作用
- [ ] 我知道 Action Chunking 如何减少累积误差
- [ ] 我能说出 ACT 的三个核心组件

### 进阶关
- [ ] 我理解 β 参数如何控制"重建 vs 先验匹配"的权衡
- [ ] 我知道 Temporal Ensemble 为什么让动作更平滑
- [ ] 我能对比 ACT 和 Diffusion Policy 在推理速度和动作多样性上的优劣
- [ ] 我理解为什么推理时从 N(0,I) 采样 z 是合理的（KL 的约束）

### 实战关
- [ ] 我能跑通 ACT 官方代码
- [ ] 我调过 β 值并观察动作多样性的变化
- [ ] 我理解 ACT 的 Encoder-Decoder 架构与标准 Transformer 的差异

## 关联笔记
- [[扩散策略]] — 同期的另一种动作生成方案，扩散模型替代 VAE
- [[行为克隆]] — 最简单的替代方案（无 VAE 无 Chunking）
- [[模仿学习]] — ACT 所属的学习范式
- [[VLA模型总览]] — ACT 在 VLA 模型家族中的位置
- [[机器人运动学]] — ACT 输出关节空间控制
