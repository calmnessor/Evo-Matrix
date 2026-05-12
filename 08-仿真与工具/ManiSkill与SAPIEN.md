# ManiSkill 与 SAPIEN

> 关节物体操作的专用仿真平台——专攻"开门、拉抽屉、旋转阀门"等需要理解物体关节的任务。

## 定位

大多数操作仿真器（MuJoCo/robosuite/PyBullet）适合 "抓取-放置" 式任务。**ManiSkill** 和底层引擎 **SAPIEN** 专为**关节物体操作**（articulated object manipulation）设计——物体有活动部件（门可以转、抽屉可以拉、按钮可以按）。

```
Isaac/MuJoCo/robosuite:  "把方块放到桌上" ← 刚体操作
ManiSkill/SAPIEN:         "打开柜门，拿出盘子" ← 关节物体操作
```

## ManiSkill

### 简介

ManiSkill = SAPIEN 引擎 + 标准操作任务 + 评估协议 + 视觉 RL 基线。

| 版本 | 年份 | 任务数 | 特点 |
|------|------|--------|------|
| ManiSkill1 | 2021 | 8 类物体交互 | 首次提出关节操作基准 |
| **ManiSkill2** | 2023 | 20+ 任务 | GPU 并行、视觉 RL、真实感渲染 |
| ManiSkill3 | 2025 | 30+ 任务 | 更真实物理、更多物体变体 |

### 安装

```bash
pip install mani-skill
# GPU 并行版本:
pip install mani-skill[gpu]
```

### 任务分类

| 类别 | 示例任务 | 关键难度 |
|------|---------|---------|
| **刚体操作** | Pick, Stack, PegInHole | 基础 |
| **关节物体** | OpenDoor, OpenDrawer, TurnFaucet | ⭐⭐ 需要理解关节轴 |
| **工具使用** | PushChair, PickUpCup | ⭐⭐⭐ 需要推断隐式约束 |
| **柔性物体** | PourWater, FillWater | ⭐⭐⭐ 流体/柔性 |
| **移动操作** | 导航+开门 | ⭐⭐⭐⭐ 复合任务 |

### 基础用法

```python
import mani_skill2.envs

# ManiSkill2 任务
env = gym.make("OpenCabinetDoor-v1",
    obs_mode="rgbd",          # RGB-D 观测
    control_mode="pd_ee_delta_pose",  # 末端增量姿态
    render_mode="cameras",
    reward_mode="dense",
)

obs, info = env.reset()
for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        break
```

### GPU 并行 (关键特性!)

```python
# ManiSkill2 的 GPU 并行——类似 Isaac Lab
# 单个 GPU 上同时运行数百个环境
import mani_skill2.envs.mpm as mpm

env = gym.make_vec("OpenCabinetDoor-v1",
    num_envs=256,         # 256 个并行环境!
    obs_mode="rgbd",
    control_mode="pd_ee_delta_pose",
)
# 每个 step 在 GPU 上批量执行
```

### VLA 评估

```python
def evaluate_vla_on_maniskill(vla_model, task_name="OpenCabinetDoor-v1",
                                num_episodes=100):
    env = gym.make(task_name,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
    )

    success_count = 0
    for ep in range(num_episodes):
        obs, _ = env.reset()
        for step in range(200):
            action = vla_model.predict({
                "rgb": obs["image"]["hand_camera"]["rgb"],
                "depth": obs["image"]["hand_camera"]["depth"],
                "proprio": obs["agent"]["qpos"],
            })
            obs, reward, terminated, truncated, info = env.step(action)
            if info["success"]:
                success_count += 1
                break

    return success_count / num_episodes
```

## SAPIEN

### 简介

SAPIEN 是 UC San Diego 开发的物理仿真引擎，专为关节物体设计。

```
SAPIEN 核心特性:
  - 原生支持关节物体 (URDF → 门/抽屉/柜子)
  - 支持 3D 模型数据集 PartNet-Mobility (2000+ 关节物体)
  - 与 ManiSkill 深度集成
```

### PartNet-Mobility 数据集

```
PartNet-Mobility = 最大的关节物体 3D 模型集
  - 46 个常见类别 (门、抽屉、柜子、水龙头、剪刀...)
  - 2000+ 个实例，每个有多变体
  - 带标注的关节轴和运动范围
  - 直接可用于 SAPIEN/ManiSkill
```

### SAPIEN 基础 API

```python
import sapien.core as sapien

# 创建引擎和场景
engine = sapien.Engine()
renderer = sapien.VulkanRenderer()
engine.set_renderer(renderer)

scene = engine.create_scene()
scene.set_timestep(1 / 240.)

# 加载关节物体
loader = scene.create_urdf_loader()
obj = loader.load("cabinet.urdf")

# 获取关节
for joint in obj.get_active_joints():
    print(f"Joint: {joint.name}, type={joint.type}")
    # type=revolute → 门铰链
    # type=prismatic → 抽屉滑轨
    print(f"  limits: {joint.get_limits()}")  # [min, max] rad/m
    joint.set_drive_target(target_angle)
```

## 与 robosuite 的对比

| | ManiSkill/SAPIEN | robosuite/MuJoCo |
|---|-----------------|------------------|
| **刚体操作** | ✅ | ✅ (专长) |
| **关节物体** | ✅✅✅ (专长) | ⚡ (需手动定义) |
| **PartNet 资产** | ✅ | ❌ |
| **GPU 并行** | ✅✅ | ❌ |
| **视觉质量** | 中等-高 | 中等 |
| **社区活跃度** | 快速增长 | 大 |
| **VLA 基准** | ManiSkill Challenge | LIBERO |

## 自检问题

- [ ] 我理解 ManiSkill 和 SAPIEN 的关系
- [ ] 我知道关节物体操作和刚体操作的区别
- [ ] 我了解 PartNet-Mobility 数据集
- [ ] 我能列举 ManiSkill2 的 5 个以上任务
- [ ] 我理解 ManiSkill GPU 并行的优势

## 关联笔记
- [[仿真与工具索引]] — 回到工具索引
- [[robosuite与模仿学习框架]] — 对比基准
- [[MuJoCo]] — 底层物理对比
- [[../../01-基础理论/机器人运动学]] — 关节类型知识
