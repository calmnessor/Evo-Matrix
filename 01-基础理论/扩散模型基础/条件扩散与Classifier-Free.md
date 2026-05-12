# 条件扩散与 Classifier-Free Guidance

> 让扩散模型"听话"——根据观测条件生成对应的动作。VLA 动作生成的核心技术。

---

## 1. 条件扩散 (Conditional Diffusion)

### 1.1 核心思想

无条件扩散：$p(x)$ — 生成"任意动作"
条件扩散：$p(x | c)$ — 生成"与观测 $c$ 匹配的动作"

其中 $c$ = 图像特征 + 语言指令 + 机器人状态。

### 1.2 实现方式

最简单的方式：把条件 $c$ 拼接到去噪网络的输入中。

```python
# 条件扩散的去噪网络
def denoise(x_t, t, condition):
    # x_t: 噪声动作 [B, act_dim]
    # condition: 条件特征 [B, cond_dim]
    # t: 时间步 [B]
    
    input = concat([x_t, condition])  # 拼接条件
    noise_pred = unet(input, t)       # 网络预测噪声
    return noise_pred
```

---

## 2. Classifier-Free Guidance (CFG)

### 2.1 动机

条件扩散的问题是：模型可能不太"听话"——在条件 $c$ 下的生成结果相对于无条件生成的差异不够大。

CFG 通过放大条件的影响来解决：

$$\epsilon_\theta^{CFG}(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

- $\epsilon_\theta(x_t, c)$：条件预测
- $\epsilon_\theta(x_t, \emptyset)$：无条件预测（把条件设为零向量/空文本）
- $w$：guidance scale，通常 $w \in [1.0, 5.0]$

### 2.2 直觉

```
当 w = 1: 就是普通条件扩散
当 w > 1: "把条件影响放大 w 倍" → 生成结果更"听话"
当 w = 0: 退化为无条件扩散

公式解读:
  无条件 + w × (有条件 - 无条件)
  = 基础方向 + 放大 × 条件修正方向
```

**权衡**：w 越大 → 生成结果越贴合条件，但多样性越低。典型 w=1.5-3.0。

### 2.3 训练时的技巧

训练时以概率 $p_{uncond}$（通常 10-20%）把条件替换为空：

```python
if random.random() < 0.15:  # 15% 概率
    condition = zero_embedding  # 替换为空条件
```

这样模型同时学会了"无条件"和"有条件"两种模式，推理时才能做 CFG。

---

## 3. Diffusion Policy

> 扩散模型 + 模仿学习 = 当前 SOTA 动作生成方案。

### 3.1 核心架构

```
观测 (图像 + 关节角)
    ↓
FiLM Encoder / CNN → 条件特征 c
    ↓
噪声动作 x_t → 1D U-Net (时间条件) → 预测噪声
    ↑
条件注入 (FiLM / Cross-Attention / concat)
    ↓
逐步去噪 → 干净动作序列 (Action Chunk)
```

### 3.2 为什么 Diffusion Policy 效果好？

| 传统 BC (MSE) | Diffusion Policy |
|---------------|-----------------|
| 学到所有演示的平均动作 | 学到的动作分布覆盖多种可能 |
| 不适合多模态动作分布 | 天然处理多模态 |
| 敏感于异常演示 | 对噪声演示更鲁棒 |
| 一步输出动作 | 多步去噪 → 渐进式动作细化 |

```
例子: "抓杯子"
  BC-MSE: 抓杯身 (平均演示位置) → 杯柄在左边也抓杯身
  Diffusion: 以一定概率抓杯柄、一定概率抓杯身 → 根据具位置自适应
```

### 3.3 关键设计选择

| 选择 | 推荐 | 原因 |
|------|------|------|
| 动作表示 | 关节角序列 | 无需 IK，无奇异性问题 |
| 预测长度 | 16-32 步 (action chunk) | 足够覆盖一个子动作 |
| 去噪步数 | 100 (训练), 16-64 (推理加速) | DDIM 加速采样 |
| 网络架构 | 1D U-Net / Transformer | 取决于动作维度 |
| 条件注入 | FiLM 或 Cross-Attention | 灵活，不限于 concat |

---

## 4. DDIM 加速采样

DDPM 采样需要 1000 步（太慢）。DDIM 允许跳步：

```python
# DDIM: 确定性跳步采样
step_schedule = [1000, 900, 800, ..., 100, 0]  # 10 步采样
# 或更激进的: [1000, 500, 0]  # 3 步
```

原理：DDIM 的逆向过程是确定性的（不加随机噪声 $z$），所以可以在大步长下保持一致。

**VLA 中的实践**：训练时用 100 步，推理时用 DDIM 压缩到 16 步，兼顾质量和速度。

---

## 关联笔记

- [[DDPM与Score-Based]] — 扩散模型基础原理
- [[../../02-VLA/扩散策略|扩散策略]] — Diffusion Policy 论文精读
- [[../../05-生成式模型/论文精读/MVGGT/MVGGT|MVGGT]] — 多视角生成，扩散应用实例
- [[../大语言模型与微调/大语言模型与微调|大语言模型与微调]] — π₀ 的 Flow Matching 替代方案
