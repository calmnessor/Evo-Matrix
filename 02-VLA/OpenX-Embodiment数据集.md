# Open X-Embodiment 数据集

> 机器人学的"ImageNet 时刻"——最大的开源机器人数据集。VLA 的泛化能力很大程度上来自 OXE 的规模和多样性。

## 为什么 OXE 重要？

在 CV 和 NLP 中，ImageNet 和 Common Crawl 推动了深度学习的革命。机器人学长期缺乏类似的大规模、多样化数据集——每个实验室用自己的机器人、自己的任务、自己的数据格式，互不相通。

OXE (Open X-Embodiment) 统一了 34 个实验室的 60+ 个数据集，建立了机器人学习的共享数据基础。

## 数据规模

- **34 个**机器人研究机构
- **60+ 个**子数据集
- **100+ 万**条轨迹
- **160+ 万**个任务
- **超过 20 种**不同的机器人平台

## 核心论文

**"Open X-Embodiment: Robotic Learning Datasets and RT-X Models"** (2023)

核心发现：在多机器人数据上联合训练的 RT-X 模型，在单个机器人上的表现**超越**了仅用该机器人数据训练的模型。这意味着 **机器人学习也存在"数据量越大→泛化越好"的 scaling 效应**。

## 数据格式 (RLDS/TensorFlow Datasets)

OXE 使用 RLDS (Reinforcement Learning Datasets) 格式，基于 TensorFlow Datasets：

```python
import tensorflow_datasets as tfds

# 加载 OXE 数据
ds = tfds.load('fractal20220817_data', split='train')

for episode in ds.take(1):
    # 每条轨迹的结构
    steps = list(episode['steps'])
    first_step = steps[0]
    
    print(first_step['observation'].keys())
    # dict_keys(['image', 'image_primary', 'image_wrist', 'state', ...])
    
    print(first_step['action'].shape)
    # (7,)  — [dx, dy, dz, dr, dp, dy, gripper]
```

### 标准数据字段

```python
trajectory = {
    "episode_metadata": {
        "episode_id": "...",
        "file_path": "...",
    },
    "steps": [
        {
            "observation": {
                "image": np.ndarray,           # [H, W, 3] 主图像
                "image_primary": np.ndarray,   # 主摄像头
                "image_wrist": np.ndarray,     # 腕部摄像头 (可选)
                "image_high": np.ndarray,      # 俯视摄像头 (可选)
                "state": np.ndarray,           # [state_dim] 关节角/末端位姿
                "state_joint": np.ndarray,     # 关节位置
                "state_ee": np.ndarray,        # 末端位姿
            },
            "action": np.ndarray,              # [action_dim] 动作
            "reward": float,                   # 奖励信号 (可选)
            "discount": float,                 # 折扣因子 (可选)
            "is_first": bool,
            "is_last": bool,
            "is_terminal": bool,
            "language_instruction": str,       # "pick up the red block"
            "language_embedding": np.ndarray,  # 预编码的语言特征
        },
        ...
    ]
}
```

### Action 格式多样性

不同数据集的动作空间不同，这是 OXE 训练的最大工程挑战：

| 类型 | 维度 | 示例数据集 | 格式 |
|------|------|-----------|------|
| 增量末端控制 | 7 | Fractal, Kuka | [dx, dy, dz, drx, dry, drz, gripper] |
| 增量末端控制 | 8 | Bridge v2 | [dx, dy, dz, drx, dry, drz, gripper_open, gripper_close] |
| 绝对关节位置 | 7+ | CMU Stretch | [θ₁, ..., θ₇] |
| 绝对末端位姿 | 6-7 | Berkeley Autolab | [x, y, z, roll, pitch, yaw, gripper] |
| 归一化范围 | — | — | 通常 [-1, 1] 或 [0, 1] |

## 重要子集详解

| 数据集 | 机构 | 机器人 | 轨迹数 | 任务类型 | 特点 |
|--------|------|--------|--------|---------|------|
| **Fractal** | Everyday Robots | Everyday Robots | ~130K | 桌面操作 | RT-1 的训练数据，规模最大 |
| **Bridge v2** | UC Berkeley | WidowX 250 | ~60K | 桌面操作 | 多视角（4 个相机），质量极高 |
| **Kuka** | Google | Kuka IIWA | ~10K | 工业操作 | 精确控制，力控 |
| **Berkeley UR5** | UC Berkeley | UR5 | ~3K | 抓取放置 | 基础操作 benchmark |
| **Stanford MaskVIT** | Stanford | Franka Panda | ~5K | 精细操作 | 科研质量，多任务 |
| **CMU Stretch** | CMU | Hello Stretch | ~2K | 移动操作 | 移动底盘+臂 |
| **DLR Sara** | DLR | Sara | ~5K | 接触式操作 | 力控，装配任务 |

## 训练 VLA 时的数据处理流水线

```python
# 完整的 OXE 数据预处理流程

def preprocess_oxe_trajectory(episode):
    """标准化一个 OXE 轨迹为统一格式"""
    processed_steps = []
    
    for step in episode['steps']:
        # 1. 统一图像尺寸和格式
        image = step['observation']['image']
        image = resize(image, (224, 224))           # 统一尺寸
        image = image / 255.0                         # 归一化到 [0, 1]
        image = normalize(image, mean, std)           # 标准化
        
        # 2. 统一动作空间 (最关键的一步!)
        action = step['action']
        action = normalize_action(action, dataset_name, action_space)
        # 不同数据集的 action 含义不同，需要分别处理
        # 方案 A: 统一到增量末端控制 [-1, 1]
        # 方案 B: 统一到关节空间 [-1, 1]
        
        # 3. 统一语言编码
        if 'language_embedding' in step:
            lang_emb = step['language_embedding']
        else:
            lang_emb = encode_language(step['language_instruction'])
            # 用一个统一的 text encoder (如 CLIP/SigLIP text branch)
        
        # 4. 构建观测字典
        obs = {
            'image_primary': image,
            'state': step['observation'].get('state'),
            'lang_emb': lang_emb,
        }
        
        processed_steps.append({'obs': obs, 'action': action})
    
    return processed_steps

# 5. 多数据集混合采样策略
def create_mixture_dataloader(datasets, weights=None):
    """
    datasets: {'fractal': ds1, 'bridge': ds2, ...}
    weights: 采样概率。None = 均匀采样
    
    常见策略:
      - 按数据量加权: weight_i = sqrt(|dataset_i|)  (缓解不平衡)
      - 均匀采样: 每个数据集等概率 (小数据集不被淹没)
      - 按任务多样性: 手动调权重
    """
    ...
```

### 关键预处理决策

| 决策 | 选项 | 推荐 |
|------|------|------|
| 图像尺寸 | 224×224 / 384×384 | OpenVLA 用 384×384 (SigLIP) |
| 动作空间 | 增量末端 / 绝对关节 / 绝对末端 | 增量末端最通用 |
| 动作归一化 | [-1,1] / [0,1] / z-score | [-1,1] 最普遍 |
| 采样策略 | 均匀 / 按 √size / 按总步数 | 按 √size 是常用折中 |
| 语言编码 | 预计算 CLIP emb / 在线计算 | 预计算省显存 |

## 数据增强

OXE 训练的常用数据增强（帮助泛化）：

```python
# 视觉增强
transforms = [
    RandomResizedCrop(224, scale=(0.8, 1.0)),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    RandomRotation(degrees=5),       # 模拟相机角度变化
    GaussianNoise(sigma=0.01),       # 模拟传感器噪声
]

# 动作增强 (小心!)
# 一般不增强动作，除非能保证物理一致性
```

## 数据加载性能优化

- **TFDS 预取**：`ds.prefetch(tf.data.AUTOTUNE)` — 必须加
- **RLDS 并行解码**：用 `tfds.builder(..., decoders={'image': tfds.decode.SkipDecoding()})` 跳过不必要的解码
- **图像预 resize**：在数据集构建阶段 resize，不在训练时（OXE 已有 224 版本）

## 自建数据与 OXE 混合训练

课题组的典型做法：

```
1. 收集 100-500 条课题组自己的演示数据
2. 80% OXE 数据 + 20% 自建数据 混合训练
3. 在自建数据上做 LoRA 微调
   → 保留 OXE 学到的泛化能力 + 适配课题组的特定任务
```

## 推荐资源

| # | 资源 | 关键词 |
|---|------|--------|
| 1 | **[OXE 官方网站](https://robotics-transformer-x.github.io/)** | 数据集列表、论文、下载 |
| 2 | **[OXE GitHub](https://github.com/google-deepmind/open_x_embodiment)** | 数据加载代码、预处理工具 |
| 3 | **[RT-X 论文](https://arxiv.org/abs/2310.08864)** | OXE + RT-X 模型的完整报告 |
| 4 | **[OpenVLA 数据处理](https://github.com/openvla/openvla)** | 看顶级 VLA 怎么处理 OXE |
| 5 | **[RLDS 文档](https://github.com/google-research/rlds)** | 数据格式详解 |

## 自检问题

### 基础关
- [ ] 我知道 OXE 包含多少数据量和来源
- [ ] 我理解 OXE 的数据格式 (RLDS) 的基本字段
- [ ] 我能列举至少 5 个 OXE 子集及其机器人和任务类型
- [ ] 我理解为什么需要统一不同数据集的动作空间

### 进阶关
- [ ] 我能解释多数据集混合采样的不同策略（均匀/按规模/按多样性）
- [ ] 我知道 OXE 训练时数据预处理的标准流程
- [ ] 我理解 RT-X 论文的核心发现（多机器人联合训练 > 单机器人训练）
- [ ] 我能说出至少 3 种标准化动作空间的方案及各自优劣

### 实战关
- [ ] 我加载过 OXE 的至少一个子集并可视化了几条轨迹
- [ ] 我实现过多数据集混合训练的 dataloader
- [ ] 我对比过不同采样策略对训练效果的影响

## 关联笔记
- [[VLA模型总览]] — OXE 是所有 VLA 基准模型的训练数据
- [[行为克隆]] — OXE 的训练方式
- [[模仿学习]] — OXE 数据来自人类演示
- [[仿真与工具]] — 仿真补充 OXE 缺少的数据
