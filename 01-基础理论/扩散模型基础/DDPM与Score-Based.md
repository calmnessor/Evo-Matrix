# DDPM 与 Score-Based 扩散模型

> 扩散模型 (Diffusion Models) 是 2024-2025 年 VLA 动作生成的主流方案之一。Diffusion Policy、π₀ 等 SOTA 方法都基于此。

---

## 1. 一句话理解

**扩散 = 逐步加噪破坏数据 + 学习逆向去噪恢复数据**

```
前向过程 (加噪, 固定): 数据 x₀ → 加噪声 → x₁ → 加噪声 → ... → x_T (纯噪声)
逆向过程 (去噪, 学习): 纯噪声 x_T → 预测噪声 → x_{T-1} → ... → x₀ (干净数据)

训练目标: 学会从任意噪声水平的 x_t 中预测出所加的噪声
推理目标: 从纯噪声开始，逐步去噪，生成新数据
```

---

## 2. DDPM (Denoising Diffusion Probabilistic Models)

### 2.1 前向过程

从真实数据 $x_0$ 开始，逐步加高斯噪声：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$$

其中 $\beta_t$ 是噪声调度（noise schedule），通常从 $\beta_1=10^{-4}$ 线性增加到 $\beta_T=0.02$。

**关键性质**：可以从 $x_0$ 一步跳到任意时刻的 $x_t$（重参数化）：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，$\epsilon \sim \mathcal{N}(0, \mathbf{I})$。

### 2.2 逆向过程

学习一个神经网络 $\epsilon_\theta(x_t, t)$ 来预测 $x_t$ 中的噪声：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

训练就是：取数据 → 加噪 → 让模型预测加了什么噪声 → 算 MSE → 反向传播。

### 2.3 采样（推理）

```python
# DDPM 采样 (简化版)
x_T = torch.randn(batch, channels, H, W)  # 纯噪声
for t in reversed(range(T)):
    z = torch.randn_like(x_T) if t > 0 else 0
    noise_pred = model(x_t, t)
    x_{t-1} = 1/sqrt(α_t) * (x_t - (1-α_t)/sqrt(1-ᾱ_t) * noise_pred) + σ_t * z
return x_0
```

---

## 3. Score-Based 视角

### 3.1 得分函数

DDPM 预测噪声 $\epsilon$，而 Score-Based 模型预测**得分函数 (Score Function)**：

$$s_\theta(x) = \nabla_x \log p(x)$$

得分函数指向"数据分布密度增加最快的方向"——跟着得分走，就能从噪声走到数据。

### 3.2 与 DDPM 的等价性

$$\epsilon_\theta(x_t, t) = -\sqrt{1-\bar{\alpha}_t} \cdot s_\theta(x_t, t)$$

预测噪声 ≈ 预测得分函数（只差一个缩放因子）。两者是等价的。

### 3.3 为什么两种视角都有用？

- **DDPM 视角**：训练目标直观（预测噪声），工程实现简单
- **Score 视角**：揭示与能量模型、Langevin 动力学的深层联系，Classifier-Free Guidance 的公式更自然

---

## 4. 在机器人中的应用位置

```
VLA 动作生成管线中的扩散模型:

观测 (图像+状态) → 条件编码器 → 条件特征
                                    ↓
噪声动作 z_T → U-Net/Transformer 去噪网络 → z_{T-1} → ... → z₀ (干净动作)
                                    ↑
                              条件特征注入
```

**为什么扩散适合动作生成？**
- 动作是多模态的（"抓左边杯子"和"抓右边杯子"都是好的）
- 扩散天然能学多模态分布（不像 MSE 回归只能学到平均动作）
- 逐步去噪 = 逐步细化动作轨迹

---

## 关联笔记

- [[条件扩散与Classifier-Free]] — 条件控制 + CFG + Diffusion Policy
- [[../深度学习基础/深度学习基础|深度学习基础]] — 训练循环、损失函数
- [[../Transformer与注意力/Transformer与注意力|Transformer与注意力]] — 扩散 Transformer (DiT)
- [[../../02-VLA/扩散策略|扩散策略]] — 扩散模型在 VLA 中的直接应用
- [[../../05-生成式模型/生成式模型方向综述|生成式模型综述]] — 生成式模型的全局视角
