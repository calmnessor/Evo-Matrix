# 🔬 Day 12: 实战项目 2 — ACT 训练与评估（8 小时）

> **口号**: "从零训练一个 ACT 策略——理解 VLA 的另一面！"  
> **目标**: 克隆 ACT 代码，在仿真环境中训练并评估  
> **为什么重要**: ACT 是 VLA 领域最经典的精细操作方案

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | ACT 代码环境搭建 | 1h | [ ] |
| 2 | 理解 ACT 代码结构 | 1.5h | [ ] |
| 3 | 配置仿真环境 | 1h | [ ] |
| 4 | 训练 ACT 模型 | 2h | [ ] |
| 5 | 评估与可视化 | 1.5h | [ ] |
| 6 | 项目总结 | 1h | [ ] |

---

## 项目目标

> 完成以下任务即为项目成功：
> 1. ✅ 成功克隆并理解 ACT 代码库
> 2. ✅ 配置仿真环境（MuJoCo/Robosuite）
> 3. ✅ 训练一个 ACT 策略（即使是简化的）
> 4. ✅ 可视化训练过程和结果
> 5. ✅ 对比 ACT 与 OpenVLA 的架构差异

---

## 模块 1: ACT 代码环境搭建（1 小时）

### 1.1 克隆与安装 — 25min

```bash
# 1. 克隆 ACT
git clone https://github.com/tonyzhaozh/act.git
cd act

# 2. 创建环境
conda create -n act python=3.8 -y
conda activate act

# 3. 安装 PyTorch
pip install torch==2.0.1 torchvision==0.15.2

# 4. 安装依赖
pip install pyquaternion  # 四元数处理
pip install pyyaml        # 配置文件
pip install rospkg        # ROS 包
pip install pexpect       # 子进程管理
pip install mujoco         # 物理引擎
pip install dm_control     # DeepMind 控制
pip install opencv-python  # 图像处理
pip install matplotlib     # 可视化
pip install ipython        # 交互式 Python
pip install einops         # 张量操作
pip install h5py           # 数据存储

# 5. 安装 robomimic（数据加载）
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
cd ..

# 6. 验证安装
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import mujoco; print('MuJoCo OK')"
python -c "import dm_control; print('dm_control OK')"
python -c "import robomimic; print('robomimic OK')"
```

### 1.2 了解 ACT 项目结构 — 20min

```
act/
├── imitate_episodes.py      ← 🔑 主训练脚本（核心入口）
├── policy.py                ← ACT 策略网络定义
├── detr/                    ← DETR Transformer 实现
│   ├── models/
│   │   ├── transformer.py   ← Transformer 编码器/解码器
│   │   └── position_encoding.py
│   └── util/
├── constants.py             ← 所有配置常量
├── utils.py                 ← 工具函数
├── visualize_episodes.py    ← 可视化脚本
├── record_sim_episodes.py   ← 数据收集脚本
└── config/
    └── default.yaml         ← 默认配置
```

### 1.3 关键配置 — 15min

打开 `constants.py`，理解这些关键配置:

```python
# ACT 的关键超参数（在 constants.py 中）

# 任务
TASK_CONFIGS = {
    'sim_transfer_cube': { ... },
    'sim_insertion': { ... },
    # 还有其他任务
}

# 训练
LR = 1e-5                    # 学习率
KL_WEIGHT = 10               # KL 散度权重
CHUNK_SIZE = 100             # 动作块大小！关键参数
BATCH_SIZE = 8               # 批量大小

# 模型
ENC_LAYERS = 4               # Transformer 编码器层数
DEC_LAYERS = 7               # Transformer 解码器层数
HIDDEN_DIM = 512             # 隐藏维度
NHEADS = 8                   # 注意力头数
DIM_FEEDFORWARD = 3200       # FFN 维度
LATENT_DIM = 32              # CVAE 隐变量维度

# 推理
TEMPORAL_AGG = True          # 时序集成
QUERY_FREQUENCY = 100        # 查询频率
```

---

## 模块 2: 理解 ACT 代码结构（1.5 小时）

### 2.1 策略网络 (`policy.py`) — 40min

```python
# ACT 策略网络的核心结构（简化阅读版）

class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 1. 视觉 Backbone: ResNet-18
        self.backbone = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # 输出: [B, 512, 15, 20] (对于 480×640 输入)

        # 2. 位置编码
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=256//2)

        # 3. Transformer 编码器（处理图像特征）
        encoder_layer = TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=3200,
            activation='relu', normalize_before=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=4)

        # 4. CVAE 编码器（编码动作 → 隐变量 z）
        self.encoder_action_proj = nn.Linear(action_dim, 512)
        self.encoder_joint_proj = nn.Linear(joint_dim, 512)
        self.latent_proj = nn.Linear(512, 32*2)  # mu + logvar

        # 5. CVAE 解码器（隐变量 z + 观测 → 动作）
        decoder_layer = TransformerDecoderLayer(
            d_model=512, nhead=8, dim_feedforward=3200,
            activation='relu', normalize_before=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=7)

        # 6. 动作输出头
        self.action_head = nn.Linear(512, action_dim)

    def forward(self, image, qpos, actions=None, is_pad=None):
        """
        训练时 actions 不为 None → 用 CVAE
        推理时 actions 为 None → 从先验采样 z
        """
        # 1. 提取图像特征
        img_feat = self.backbone(image)  # [B, 512, H, W]
        img_feat = self.pos_encoder(img_feat)  # 加位置编码
        img_feat = img_feat.flatten(2).permute(2, 0, 1)  # [N, B, 512]

        # 2. 编码观测（图像 + 关节状态）
        qpos_emb = self.encoder_joint_proj(qpos)  # [B, 512]
        encoder_input = torch.cat([img_feat, qpos_emb.unsqueeze(0)], dim=0)
        encoder_output = self.encoder(encoder_input)

        if actions is not None:
            # === 训练模式: CVAE ===
            # 编码动作
            action_emb = self.encoder_action_proj(actions)
            # ... 通过 encoder 得到 mu, logvar
            # 重参数化采样 z
            # 用 z + encoder_output 通过 decoder 重建动作
            # 返回: pred_actions, mu, logvar
            ...
        else:
            # === 推理模式 ===
            # 从 N(0, I) 采样 z
            # 用 z + encoder_output 通过 decoder 生成动作
            # 返回: pred_actions
            ...
```

### 2.2 训练循环 (`imitate_episodes.py`) — 30min

```python
# ACT 训练流程（简化理解）

def main(args):
    # 1. 加载数据
    dataset = EpisodicDataset(...)
    dataloader = DataLoader(dataset, ...)

    # 2. 创建策略
    policy = ACTPolicy(args).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

    # 3. 训练循环
    for epoch in range(num_epochs):
        for batch in dataloader:
            # 数据: image, qpos, action, is_pad
            image = batch['image'].to(device)
            qpos = batch['qpos'].to(device)
            action = batch['action'].to(device)

            # 前向
            pred_action, mu, logvar = policy(image, qpos, action)

            # 计算损失
            recon_loss = F.l1_loss(pred_action, action)
            kl_loss = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum()
            loss = recon_loss + KL_WEIGHT * kl_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            # 记录
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}: "
                      f"Loss={loss:.4f}, Recon={recon_loss:.4f}, KL={kl_loss:.4f}")

    # 4. 保存模型
    torch.save(policy.state_dict(), 'policy_best.pt')
```

### 2.3 理解数据格式 — 20min

```python
# ACT 使用的数据格式（每个 episode 是一个 HDF5 文件）

# episode_0000.hdf5 包含:
# /observations/
#   /images/             ← 图像序列 (来自多个相机)
#     /camera_1
#     /camera_2
#   /qpos                ← 关节位置序列 [T, joint_dim]
#   /qvel                ← 关节速度序列
# /action                ← 动作序列 [T, action_dim]
#
# 数据收集方式:
#   1. 人用遥操作设备控制机械臂
#   2. 记录所有传感器数据
#   3. 保存为 HDF5 格式

# 读取一条轨迹
import h5py

with h5py.File('episode_0.hdf5', 'r') as f:
    print("Keys:", list(f.keys()))
    print("Observation keys:", list(f['observations'].keys()))
    images = f['observations/images/camera_1'][:]  # [T, H, W, C]
    actions = f['action'][:]                       # [T, action_dim]
    print(f"Images shape: {images.shape}")
    print(f"Actions shape: {actions.shape}")
```

---

## 模块 3: 配置仿真环境（1 小时）

### 3.1 使用简化仿真环境 — 40min

由于 ACT 原始仿真环境需要特定的硬件和大量配置，我们用一个简化的 Gym 环境来理解整个流程:

```python
"""简化版 ACT 训练环境"""
import gymnasium as gym
import numpy as np
import torch
from collections import deque

class SimpleACTEnv:
    """
    简化的仿真环境（用 Gym 的 Mujoco 环境）
    展示 ACT 训练所需的数据格式
    """
    def __init__(self, env_name="HalfCheetah-v4"):
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

    def collect_episode(self, policy=None):
        """
        收集一条 episode 数据
        policy: 如果有专家策略则使用，否则随机
        """
        obs, _ = self.env.reset()
        done = False

        images = []
        states = []
        actions = []

        while not done:
            # 渲染图像
            img = self.env.render()  # [H, W, C]

            # 选择动作
            if policy is not None:
                action = policy(obs)
            else:
                action = self.env.action_space.sample()

            next_obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            images.append(img)
            states.append(obs)
            actions.append(action)

            obs = next_obs

        return {
            "images": np.array(images),      # [T, H, W, C]
            "states": np.array(states),      # [T, obs_dim]
            "actions": np.array(actions),    # [T, action_dim]
        }

    def collect_dataset(self, n_episodes=50, policy=None):
        """收集多条 episode"""
        dataset = []
        for i in range(n_episodes):
            episode = self.collect_episode(policy)
            dataset.append(episode)

            # 保存为 HDF5（模拟 ACT 训练数据格式）
            import h5py
            with h5py.File(f'episode_{i:04d}.hdf5', 'w') as f:
                f.create_dataset('observations/images/camera_1',
                               data=episode['images'])
                f.create_dataset('observations/qpos',
                               data=episode['states'])
                f.create_dataset('action',
                               data=episode['actions'])

            if i % 10 == 0:
                print(f"Collected {i+1}/{n_episodes} episodes")

        return dataset


# 测试收集一条数据
env = SimpleACTEnv("HalfCheetah-v4")
episode = env.collect_episode()
print(f"Episode length: {len(episode['actions'])}")
print(f"Image shape: {episode['images'].shape}")
print(f"State shape: {episode['states'].shape}")
print(f"Action shape: {episode['actions'].shape}")
```

### 3.2 可视化一条轨迹 — 20min

```python
import matplotlib.pyplot as plt

def visualize_episode(episode):
    """可视化一条轨迹"""
    T = len(episode['actions'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 第一帧
    axes[0, 0].imshow(episode['images'][0])
    axes[0, 0].set_title("First Frame")
    axes[0, 0].axis('off')

    # 最后一帧
    axes[0, 1].imshow(episode['images'][-1])
    axes[0, 1].set_title("Last Frame")
    axes[0, 1].axis('off')

    # 状态变化
    for i in range(min(6, episode['states'].shape[1])):
        axes[1, 0].plot(episode['states'][:, i], label=f'dim_{i}')
    axes[1, 0].set_title("State over time")
    axes[1, 0].legend(fontsize=6)

    # 动作变化
    for i in range(min(6, episode['actions'].shape[1])):
        axes[1, 1].plot(episode['actions'][:, i], label=f'dim_{i}')
    axes[1, 1].set_title("Action over time")
    axes[1, 1].legend(fontsize=6)

    plt.tight_layout()
    plt.savefig('episode_visualization.png', dpi=150)
    print("Saved to episode_visualization.png")

# visualize_episode(episode)
```

---

## 模块 4: 训练 ACT 模型（2 小时）

### 4.1 简化版 ACT 训练 — 60min

创建 `simple_act_train.py`:

```python
"""简化 ACT 训练脚本"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# ===== 简化版 ACT 模型 =====
class SimpleACT(nn.Module):
    """
    简化版 ACT:
    - 用状态向量代替图像（训练更快）
    - 保留 CVAE 架构
    - 保留 Action Chunking
    """
    def __init__(self, state_dim, action_dim, chunk_size=50, latent_dim=32):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # CVAE 编码器: q(z | state, action_chunk)
        self.encoder = nn.Sequential(
            nn.Linear(256 + chunk_size * action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2),  # mu + logvar
        )

        # CVAE 解码器: p(action_chunk | state, z)
        self.decoder = nn.Sequential(
            nn.Linear(256 + latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, chunk_size * action_dim),
        )

    def forward(self, state, action_chunk=None):
        """
        state: [B, state_dim]
        action_chunk: [B, chunk_size, action_dim], 训练时提供
        """
        B = state.shape[0]

        # 编码状态
        state_feat = self.state_encoder(state)  # [B, 256]

        if action_chunk is not None:
            # === 训练模式 ===
            action_flat = action_chunk.view(B, -1)  # [B, K*D]

            # 编码器
            enc_input = torch.cat([state_feat, action_flat], dim=-1)
            enc_out = self.encoder(enc_input)

            mu = enc_out[:, :self.latent_dim]
            logvar = enc_out[:, self.latent_dim:]

            # 重参数化
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps

            # 解码器
            dec_input = torch.cat([state_feat, z], dim=-1)
            pred_flat = self.decoder(dec_input)
            pred_actions = pred_flat.view(B, self.chunk_size, self.action_dim)

            # 损失
            recon_loss = F.l1_loss(pred_actions, action_chunk)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

            return pred_actions, mu, logvar, recon_loss, kl_loss
        else:
            # === 推理模式: 从先验采样 ===
            z = torch.randn(B, self.latent_dim).to(state.device)
            dec_input = torch.cat([state_feat, z], dim=-1)
            pred_flat = self.decoder(dec_input)
            return pred_flat.view(B, self.chunk_size, self.action_dim)


# ===== 准备数据 =====
def prepare_chunked_data(episodes, chunk_size=50):
    """
    从原始轨迹中提取 (state, action_chunk) 对
    """
    states = []
    chunks = []

    for ep in episodes:
        T = len(ep['actions'])
        for t in range(T - chunk_size):
            states.append(ep['states'][t])
            chunks.append(ep['actions'][t:t+chunk_size])

    return (
        torch.FloatTensor(np.array(states)),
        torch.FloatTensor(np.array(chunks)),
    )


# ===== 训练 =====
def train_simple_act(env_name="HalfCheetah-v4", n_episodes=50, n_epochs=200):
    import gymnasium as gym

    # 1. 收集数据
    print("Collecting data...")
    env = SimpleACTEnv(env_name)
    episodes = env.collect_dataset(n_episodes)

    # 2. 准备数据
    states, action_chunks = prepare_chunked_data(episodes, chunk_size=50)
    print(f"Training samples: {len(states)}")
    dataset = torch.utils.data.TensorDataset(states, action_chunks)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    state_dim = states.shape[1]
    action_dim = action_chunks.shape[2]

    # 3. 创建模型
    model = SimpleACT(state_dim, action_dim, chunk_size=50)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # 4. 训练
    metrics = {"recon": [], "kl": [], "total": []}
    KL_WEIGHT = 10.0

    for epoch in range(n_epochs):
        epoch_recon, epoch_kl, epoch_total = 0, 0, 0
        n_batches = 0

        for batch_states, batch_chunks in loader:
            pred, mu, logvar, recon_loss, kl_loss = model(batch_states, batch_chunks)
            total_loss = recon_loss + KL_WEIGHT * kl_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            epoch_total += total_loss.item()
            n_batches += 1

        scheduler.step()

        metrics["recon"].append(epoch_recon / n_batches)
        metrics["kl"].append(epoch_kl / n_batches)
        metrics["total"].append(epoch_total / n_batches)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Recon={metrics['recon'][-1]:.4f}, "
                  f"KL={metrics['kl'][-1]:.4f}, Total={metrics['total'][-1]:.4f}")

    # 5. 保存
    torch.save(model.state_dict(), "simple_act_model.pt")

    # 6. 画训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(metrics["recon"]); axes[0].set_title("Reconstruction Loss")
    axes[1].plot(metrics["kl"]); axes[1].set_title("KL Divergence")
    axes[2].plot(metrics["total"]); axes[2].set_title("Total Loss")
    plt.tight_layout()
    plt.savefig("act_training_curves.png", dpi=150)
    print("Training curves saved to act_training_curves.png")

    return model, metrics

# 运行训练
if __name__ == "__main__":
    model, metrics = train_simple_act(n_episodes=30, n_epochs=200)
    print("Training complete!")
```

### 4.2 监控训练 — 30min

```python
"""训练监控工具"""
import time

class TrainingMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.losses = []

    def update(self, loss, epoch, n_epochs):
        self.losses.append(loss)
        elapsed = time.time() - self.start_time
        eta = (elapsed / (epoch + 1)) * (n_epochs - epoch - 1)

        print(f"Epoch [{epoch+1}/{n_epochs}] "
              f"Loss: {loss:.4f} | "
              f"Elapsed: {elapsed/60:.1f}min | "
              f"ETA: {eta/60:.1f}min")

    def check_health(self):
        """检查训练是否健康"""
        if len(self.losses) < 10:
            return

        recent_5 = self.losses[-5:]
        earlier_5 = self.losses[-10:-5]

        if np.mean(recent_5) > np.mean(earlier_5) * 1.2:
            print("⚠️  WARNING: Loss increasing! Consider reducing LR")
        elif np.std(recent_5) > np.mean(recent_5) * 0.5:
            print("⚠️  WARNING: Loss oscillating! Consider smaller LR or larger batch")
        else:
            print("✅ Training looks healthy")

# 使用
# monitor = TrainingMonitor()
# for epoch in range(n_epochs):
#     loss = train_one_epoch(...)
#     monitor.update(loss, epoch, n_epochs)
#     if epoch % 10 == 0:
#         monitor.check_health()
```

---

## 模块 5: 评估与可视化（1.5 小时）

### 5.1 动作生成评估 — 30min

```python
def evaluate_act(model, test_episodes):
    """评估 ACT 策略"""
    model.eval()
    all_errors = []

    for episode in test_episodes:
        T = len(episode['actions'])
        state = torch.FloatTensor(episode['states'][0:1])  # [1, D]

        # 预测动作 chunk
        with torch.no_grad():
            pred_chunk = model(state)  # [1, K, D] 推理模式

        # 取第一个动作
        pred_action = pred_chunk[0, 0].numpy()
        true_action = episode['actions'][0]

        error = np.mean((pred_action - true_action) ** 2)
        all_errors.append(error)

    avg_error = np.mean(all_errors)
    print(f"Average MSE on test episodes: {avg_error:.6f}")
    return avg_error
```

### 5.2 动作序列可视化 — 30min

```python
def visualize_action_prediction(model, episode):
    """可视化预测动作 vs 真实动作"""
    state = torch.FloatTensor(episode['states'][0:1])

    with torch.no_grad():
        pred_chunk = model(state)[0].numpy()  # [K, D]
        true_chunk = episode['actions'][:50]   # [K, D]

    K = min(50, len(pred_chunk))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i in range(min(6, pred_chunk.shape[1])):
        ax = axes[i // 3, i % 3]
        ax.plot(range(K), pred_chunk[:K, i], 'r-', label='Predicted', alpha=0.8)
        ax.plot(range(K), true_chunk[:K, i], 'b--', label='Ground Truth', alpha=0.8)
        ax.set_title(f'Action Dim {i}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('ACT: Predicted vs Ground Truth Action Sequences')
    plt.tight_layout()
    plt.savefig('act_action_comparison.png', dpi=150)
    print("Action comparison saved to act_action_comparison.png")
```

### 5.3 隐空间可视化 — 30min

```python
def visualize_latent_space(model, episodes):
    """可视化 CVAE 的隐空间"""
    all_mu = []
    all_labels = []

    model.eval()
    for i, episode in enumerate(episodes):
        for t in range(0, len(episode['actions']) - 50, 10):
            state = torch.FloatTensor(episode['states'][t:t+1])
            chunk = torch.FloatTensor(episode['actions'][t:t+50]).unsqueeze(0)

            with torch.no_grad():
                _, mu, logvar, _, _ = model(state, chunk)

            all_mu.append(mu[0].numpy())
            all_labels.append(i)

    all_mu = np.array(all_mu)  # [N, latent_dim]

    # PCA 降维到 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(all_mu)

    # 画图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mu_2d[:, 0], mu_2d[:, 1],
                         c=all_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Episode')
    plt.title('ACT Latent Space (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.savefig('act_latent_space.png', dpi=150)
    print("Latent space visualization saved to act_latent_space.png")
```

---

## 📋 项目总结（1 小时）

### ACT vs OpenVLA 最终对比

| 维度 | OpenVLA | ACT |
|------|---------|-----|
| 架构 | VLM (SigLIP + Llama) | CVAE + Transformer |
| 参数规模 | 7B | ~100M |
| 动作表示 | 自回归离散 token | 连续值 Chunk |
| 推理速度 | 慢 (需生成) | 快 (单次 forward) |
| 语言理解 | 强 (LLM) | 弱 (简单编码) |
| 视觉理解 | 强 (VLM pretrain) | 可 (ResNet) |
| 精细操作 | 一般 | 强 |
| 部署难度 | 高 (大 GPU) | 低 |
| 开源 | ✅ | ✅ |

### 面试可以这么说

```
"我基于 ACT 的 CVAE 架构训练了一个简化版策略:
 - 实现了 Action Chunking (K=50) 以缓解分布漂移
 - 使用 KL 散度正则化隐空间
 - 理解了 CVAE 在 VLA 中的作用: 捕捉动作的多模态分布
 - 对比了 ACT (CVAE) 和 OpenVLA (自回归 VLM) 的适用场景"
```

### 今日产出清单

- [ ] 成功克隆 ACT 代码库
- [ ] 理解了 ACT 的 CVAE 架构
- [ ] 搭建了仿真环境
- [ ] 训练了一个简化版 ACT（loss 正常下降）
- [ ] 可视化了预测动作 vs 真实动作
- [ ] 总结了 ACT vs OpenVLA

---

> ✅ **完成打卡**: 两个实战项目完成！明天总复习！
>
> 🔜 **明日预告**: Day 13 — 费曼复习日，知识体系完整梳理！
