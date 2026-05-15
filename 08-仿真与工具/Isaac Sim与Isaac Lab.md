# Isaac Sim 与 Isaac Lab

> NVIDIA 的 RTX 光追仿真平台——具身智能研究的大规模 GPU 并行仿真 + RL 训练框架。

## 定位

```
Isaac Sim = 仿真平台 (类似 MuJoCo + Blender 的结合)
  - 基于 NVIDIA Omniverse / USD
  - 光追渲染 (RTX)
  - 物理仿真 (PhysX 5)
  - Python API + GUI

Isaac Lab = RL 训练框架 (跑在 Isaac Sim 之上)
  - 类似 gymnasium 但 GPU 加速
  - 大规模并行环境 (数千个环境同时运行)
  - 内置 PPO/SAC/Dreamer 等算法
  - 为 VLA / 具身智能设计的工作流
```

## 为什么用 Isaac Sim？

### 与 MuJoCo 的核心差异

| | MuJoCo | Isaac Sim |
|---|--------|-----------|
| **渲染** | 基础 OpenGL | RTX 光追（照片级） |
| **并行** | 多进程 CPU | GPU 并行（数千环境） |
| **视觉策略** | 不推荐 | ✅ 核心优势 |
| **域随机化** | 手动脚本 | 内置可视化工具 |
| **SDF/USD** | 不支持 | 原生支持 |
| **ROS 集成** | 有限 | 原生 ROS 2 bridge |
| **硬件需求** | CPU 即可 | NVIDIA GPU (RTX) |
| **安装** | `pip install mujoco` | Docker / Omniverse Launcher |

**选择指南**:
- 需要照片级渲染训练视觉策略 → Isaac Sim
- 需要数千个环境并行 RL → Isaac Sim + Isaac Lab
- 只需要运动学/动力学验证 → MuJoCo
- 快速原型 → MuJoCo / PyBullet

## Isaac Sim 基础

### 安装

```bash
# 方法 1: Docker (推荐)
docker pull nvcr.io/nvidia/isaac-sim:4.2.0
docker run --gpus all -it nvcr.io/nvidia/isaac-sim:4.2.0

# 方法 2: Omniverse Launcher (Windows/Linux GUI)
# 下载 Omniverse Launcher → 安装 Isaac Sim
```

### Python API

```python
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid
import numpy as np

# 创建世界
world = World(physics_dt=1/60, rendering_dt=1/60)

# 加载机器人
from omni.isaac.franka import Franka
franka = Franka(prim_path="/World/Franka", name="franka")
world.scene.add(franka)

# 添加物体
cube = DynamicCuboid(
    prim_path="/World/Cube",
    name="cube",
    position=np.array([0.5, 0, 0.05]),
    size=0.05,
    color=np.array([1, 0, 0]),
)
world.scene.add(cube)

# 重置并仿真
world.reset()
for step in range(1000):
    # 获取观测
    joint_pos = franka.get_joint_positions()

    # 设置动作
    franka.apply_action(np.zeros(7))

    # 步进
    world.step(render=True)

    # 获取渲染图像 (光追!)
    if step % 100 == 0:
        rgb = world.get_rendering()  # 照片级图像
```

### USD (Universal Scene Description)

Isaac Sim 使用 USD 作为场景格式，比 MJCF 更强大：

```
USD 的优势:
  - 支持层级结构、材质、动画、变体
  - 工业标准 (Pixar 开发)
  - 支持引用和组合 (大型场景的模块化)
  - 丰富的工具生态 (Blender/Maya/Houdini 都可导出 USD)
```

```python
from omni.isaac.core.utils.stage import add_reference_to_stage

# 加载 USD 资产
add_reference_to_stage(
    usd_path="/path/to/kitchen_scene.usd",
    prim_path="/World/Kitchen"
)
```

## Isaac Lab

### 核心理念

```
传统 RL:  CPU × N 进程 → N 个环境 (N ≤ 16-32)
Isaac Lab: GPU × M 个流 → M 个环境 (M ≥ 4096!)

加速比: 100-1000× (取决于任务复杂度)
```

### 安装

```bash
# Isaac Lab 作为独立包安装
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

### 创建 RL 环境

```python
# Isaac Lab 使用类似 gymnasium 的接口
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import ObservationManager, RewardManager

class MyManipulationEnv(ManagerBasedRLEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 观测: 图像 + 本体感知
        self.obs_manager = ObservationManager({
            "policy": {
                "image": {"func": self._get_image},
                "joint_pos": {"func": self._get_joint_pos},
                "ee_pose": {"func": self._get_ee_pose},
            }
        })
        # 奖励: 距离目标 + 抓取成功
        self.reward_manager = RewardManager({
            "dist_to_goal": {"func": self._reward_dist, "weight": 1.0},
            "grasp_success": {"func": self._reward_grasp, "weight": 10.0},
        })

    def _get_image(self):
        return self.sensors["camera"].data.output["rgb"]

    def _reward_dist(self):
        ee_pos = self.robot.data.ee_pos
        goal_pos = self.goal_pos
        return -np.linalg.norm(ee_pos - goal_pos)
```

### 训练脚本

```python
from omni.isaac.lab.utils import train_rl

# Isaac Lab 内置 PPO 训练器
train_rl(
    env_cfg=MyManipulationEnvCfg(),
    agent_cfg="rl_games_ppo_cfg.yaml",  # PPO 配置
    num_envs=4096,                       # 4096 个并行环境!
    max_iterations=10000,
    log_dir="./logs/manipulation",
)
```

### VLA 评估

```python
# Isaac Lab 中评估 VLA 策略
def evaluate_vla_in_isaac_lab(vla_model, env, num_episodes=100):
    obs = env.reset()
    successes = 0

    for ep in range(num_episodes):
        for step in range(200):  # max steps per episode
            # VLA 推理
            with torch.no_grad():
                action = vla_model.predict({
                    "image": obs["image"].unsqueeze(0),
                    "proprio": obs["joint_pos"].unsqueeze(0),
                })

            # 环境步进
            obs, reward, done, info = env.step(action)

            if "success" in info and info["success"]:
                successes += 1
                break

        obs = env.reset()

    return successes / num_episodes
```

## 关键工作流

### 工作流 1: 大规模视觉 RL 训练

```
Blender/USD 资产 → Isaac Sim 场景搭建
    → Domain Randomization (纹理/光照/物理)
    → Isaac Lab 4096 环境并行
    → PPO/RL 训练
    → 策略部署 (Sim2Real)
```

### 工作流 2: VLA 数据生成

```
脚本策略 (或 VLA) 在 Isaac Sim 中运行
    → 照片级渲染图像 + 分割 mask + 深度图
    → 自动收集 (image, state, action) 数据
    → 导出为 RLDS/HuggingFace Dataset
    → 训练 VLA
```

### 工作流 3: Sim2Real

```
Isaac Sim 中域随机化训练
    → RTX 渲染 (接近真实照片的质量)
    → 策略成功率高
    → 迁移到真实机器人
    ↓
如果真机性能不够:
    → 增加域随机化范围
    → 增加场景多样性
    → 添加真实纹理/光照
```

## 资源需求与配置

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 2060 (6GB) | RTX 4090 (24GB) |
| RAM | 16 GB | 64 GB |
| 存储 | 50 GB | 200 GB (USD 资产) |
| OS | Ubuntu 22.04 | Ubuntu 22.04 / Windows 11 |

## 自检问题

### 基础关
- [ ] 我理解 Isaac Sim 和 Isaac Lab 的定位差异
- [ ] 我知道 Isaac Sim vs MuJoCo 的选择标准
- [ ] 我能用 Python API 加载机器人并控制关节

### 进阶关
- [ ] 我了解 USD 格式和基本操作
- [ ] 我能配置 Domain Randomization
- [ ] 我理解 GPU 并行仿真 (4096 envs) 的原理和收益

### 实战关
- [ ] 我在 Isaac Sim 中创建过自定义操作场景
- [ ] 我跑通过 Isaac Lab 的 PPO 训练
- [ ] 我在 Isaac Sim 中评估过 VLA 模型
- [ ] 我做过 Isaac Sim → 真机的 Sim2Real 迁移

## 关联笔记
- [[仿真与工具索引]] — 回到工具索引
- [[MuJoCo]] — 对比：CPU 物理引擎
- [[仿真数据与Domain Randomization]] — Sim2Real 关键技术
- [[../../01-基础理论/强化学习]] — RL 训练理论基础
