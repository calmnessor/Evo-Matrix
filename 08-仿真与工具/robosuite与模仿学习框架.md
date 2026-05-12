# robosuite 与模仿学习框架

> VLA 评估和模仿学习训练的标准化工具链——robosuite (仿真) + robomimic (算法) + LIBERO (基准)。

## 生态概览

```
robosuite:    仿真环境 (MuJoCo 实现，10+ 标准操作任务)
robomimic:    模仿学习算法库 (BC, IRIS, Diffusion Policy 等)
LIBERO:       VLA 专用基准 (130 任务，基于 robosuite)
robosuite-vla: VLA 评估封装
```

三者关系：
```
LIBERO (定义 130 任务和评估协议)
    ↓ 基于
robosuite (提供任务仿真器)
    ↓ 使用算法
robomimic (提供 BC/IRIS/Diffusion 等实现)
```

## robosuite — 标准化操作仿真

### 为什么用 robosuite？

robosuite 是最广泛使用的**机器人操作仿真基准**之一，因为它：

1. **任务标准化**：10 个经典操作任务，参数和评估协议已固化
2. **数据收集**：内置多种输入设备（SpaceMouse、键盘、VR 手柄）
3. **MuJoCo 后端**：继承了 MuJoCo 的物理精度和速度
4. **VLA 友好**：支持语言指令、多视角渲染、本体感知

### 安装

```bash
pip install robosuite
```

### 任务一览

| 任务 | 描述 | 难度 | VLA 常用 |
|------|------|------|---------|
| **Lift** | 抓方块并提起 | ⭐ | ✅ |
| **Stack** | 方块 A 叠到方块 B 上 | ⭐⭐ | ✅ |
| **Nut Assembly** | 螺母套到杆上 | ⭐⭐⭐ | ✅ |
| **Pick and Place** | 抓取并放置到指定区域 | ⭐⭐ | ✅ |
| **Door Opening** | 打开门 | ⭐⭐⭐ | ✅ |
| **Wipe** | 用抹布擦拭桌面 | ⭐⭐ | ✅ |
| **Peg in Hole** | 插孔（精密装配） | ⭐⭐⭐ | ✅ |
| **Tool Hang** | 工具挂到挂钩上 | ⭐⭐⭐⭐ | ✅ |
| **Coffee Push** | 推咖啡杯到目标 | ⭐⭐ | — |
| **Coffee Pull** | 拉咖啡杯到目标 | ⭐⭐ | — |

### 基础使用

```python
import robosuite
import numpy as np

# 创建环境
env = robosuite.make(
    "Lift",
    robots=["Panda"],           # 机器人型号
    controller_configs={        # 控制器
        "type": "OSC_POSE",     # 操作空间控制 (末端位姿)
        "interpolation": "linear",
    },
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=480,
    camera_widths=640,
    reward_shaping=True,
)

# 重置并运行
obs = env.reset()
for step in range(500):
    # 产生随机动作 [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
    action = np.random.randn(7) * 0.1
    action[-1] = np.random.binomial(1, 0.5)  # 夹爪: 0 或 1

    obs, reward, done, info = env.step(action)

    # obs 包含:
    #   obs["agentview_image"] — 第三人称视角 (480, 640, 3)
    #   obs["robot0_eye_in_hand_image"] — 手眼相机
    #   obs["robot0_proprio-state"] — 本体感知 [cos/sin 关节角 + 末端位姿]
    #   obs["object-state"] — 物体位姿 (用于 reward 计算)

    if done:
        break
```

### OSC_POSE 控制器

robosuite 默认使用操作空间控制 (Operational Space Control)，直接输出末端位姿变化：

```python
# action 空间 (OSC_POSE):
# [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
#   ↑ 位置变化 (m)    ↑ 旋转变化 (rad)    ↑ -1(开) ~ 1(关)

# 也可以用 JOINT_POSITION 关节位置控制:
env = robosuite.make("Lift", controller_configs={"type": "JOINT_POSITION"})
# action: 7 个关节目标角度
```

### 数据收集

```python
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import input2action
import h5py

# robosuite 内置遥操作数据收集
# 支持 SpaceMouse / 键盘 / VR 控制器
env = robosuite.make(
    "Lift",
    has_offscreen_renderer=True,
    use_camera_obs=True,
)

# 收集一个 episode
observations = []
for step in range(500):
    # 从输入设备读取动作 (人工控制)
    action = input2action(device, robot="Panda")
    obs, reward, done, _ = env.step(action)
    observations.append({
        "obs": obs,
        "action": action,
    })
    if done:
        break

# 保存为 HDF5
with h5py.File("demo.h5", "w") as f:
    for key in observations[0]:
        f.create_dataset(key, data=np.stack([o[key] for o in observations]))
```

## robomimic — 模仿学习算法库

robomimic 提供标准化的模仿学习算法实现和评估协议。

```bash
pip install robomimic
```

### 支持的算法

| 算法 | 类型 | 论文 |
|------|------|------|
| BC | 监督学习 | — |
| BC-RNN | 循环 BC | — |
| BC-Transformer | Transformer BC | — |
| IRIS | BC + 数据增强 | [arXiv] |
| Diffusion Policy | 扩散策略 | [2303.04137](https://arxiv.org/abs/2303.04137) |
| BC-ACT | ACT 风格 | [2304.13705](https://arxiv.org/abs/2304.13705) |

### 训练

```python
import robomimic
from robomimic.config import Config
from robomimic.scripts.train import train

# 配置训练
config = Config("bc_config.json")
config.train.data = "path/to/demos.hdf5"
config.train.output_dir = "./bc_checkpoints"
config.algo_name = "bc"

# 训练
train(config)
```

## LIBERO — VLA 标准基准

LIBERO 是专门为 VLA 设计的基准测试，已成为事实上的 VLA 评估标准。

### 基准组成

```
LIBERO-100:    100 任务 (50 训练场景 + 50 未见场景)
LIBERO-90:     90 长周期任务
LIBERO-10:     10 任务快速原型
LIBERO-OBJECT:  物体泛化 (训练中未见的物体)
LIBERO-GOAL:    目标泛化 (新的目标位置)
LIBERO-SPATIAL: 空间泛化 (新的物体摆放)
```

### 安装

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

### 运行 LIBERO 评估

```python
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np

# 加载 LIBERO-100 基准
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_100"]

# 遍历所有任务
num_success = 0
for task_id in range(100):
    task = task_suite.get_task(task_id)

    # 创建环境
    env_args = {
        "task_name": task.name,
        "bddl_file_name": f"{task.problem_folder}/{task.bddl_file}",
        "camera_heights": 224,
        "camera_widths": 224,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
    }
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    env.set_init_state(task.init_states[0])

    # VLA 推理
    for step in range(500):
        image = obs["agentview_image"]
        proprio = obs["robot0_proprio-state"]
        lang = task.language  # 语言指令

        action = vla_model.predict(image, proprio, lang)

        obs, reward, done, info = env.step(action)
        if done:
            num_success += 1
            break

print(f"LIBERO-100 Success Rate: {num_success / 100:.2%}")
```

### LIBERO 评估报告模板

```
模型: π₀.₅-KI
基准: LIBERO-100
结果:
  ├─ LIBERO-OBJECT:  72%  (物体泛化)
  ├─ LIBERO-GOAL:    68%  (目标泛化)
  ├─ LIBERO-SPATIAL: 75%  (空间泛化)
  └─ LIBERO-100 avg: 71.7%
```

## 自检问题

### 基础关
- [ ] 我理解 robosuite、robomimic、LIBERO 三者的关系
- [ ] 我能列举 robosuite 的 5 个以上标准任务
- [ ] 我理解 OSC_POSE 和 JOINT_POSITION 控制方式

### 进阶关
- [ ] 我能用 robomimic 训练 BC/Diffusion Policy
- [ ] 我理解 LIBERO 的四个泛化维度
- [ ] 我能运行完整的 LIBERO-100 评估

### 实战关
- [ ] 我在 robosuite 中收集过遥操作演示数据
- [ ] 我对比过不同算法在 LIBERO 上的性能
- [ ] 我复现过 OpenVLA/π₀.₅ 在 LIBERO 上的结果

## 关联笔记
- [[仿真与工具索引]] — 回到工具索引
- [[MuJoCo]] — robosuite 的物理后端
- [[CALVIN与RLBench]] — 其他 VLA 基准对比
- [[../../02-VLA/行为克隆]] — robomimic 算法的理论基础
- [[../../02-VLA/VLA模型总览]] — 各模型在 LIBERO 上的表现
