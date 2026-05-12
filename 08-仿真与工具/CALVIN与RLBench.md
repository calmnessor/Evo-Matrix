# CALVIN 与 RLBench

> 两个互补的 VLA 基准——CALVIN 测语言条件长周期任务泛化，RLBench 测 100 任务大规模多任务学习。

## CALVIN — 语言条件长周期任务

### 简介

CALVIN (Composing Actions from Language and Vision) 是 ETH Zurich 和 MPI 联合开发的基准。核心设计：**多步语言指令链 + 跨环境泛化**。

```
CALVIN 核心挑战:
  指令: "打开抽屉 → 拿起蓝色积木 → 放入抽屉"
  环境: 训练在 A/B/C，测试在 D（完全不同的环境布局）
  评价: 连续成功完成的指令步数
```

### 环境设计

```
4 个环境 (A/B/C/D):
  - 每个有 1 个 Franka Panda 机械臂
  - 1 个抽屉
  - 1 个滑动门
  - 1 个 LED (可旋转)
  - 1 个开关 (可按)
  - 1 个按钮 (可按，有灯)
  - 红/蓝/粉三色积木

训练: A + B + C 的演示数据
测试: D → 完全没有见过的环境!
```

### 安装

```bash
git clone https://github.com/mees/calvin.git
cd calvin
pip install -e .
```

### 评估协议

```python
from calvin_env.envs.play_table_env import PlayTableEnv
import numpy as np

env = PlayTableEnv(env="D")  # 测试环境 D
obs = env.reset()

# 多步指令
instructions = [
    "open the drawer",
    "pick up the blue block",
    "put the blue block in the drawer",
]

completed_steps = 0
for instruction in instructions:
    # 执行当前指令 (VLA 控制)
    for step in range(200):
        action = vla_model.predict(obs["rgb"], instruction)
        obs, _, done, info = env.step(action)
        if info.get("subtask_complete"):
            completed_steps += 1
            break

print(f"Completed {completed_steps}/{len(instructions)} steps")
```

### 评价指标

```
标准指标: 平均完成步数 (average steps completed)
  - 0 步: 第一个指令就失败了
  - 3 步: 连续完成了所有 3 步 — 满分!
  - 1.5 步: 平均有时能完成 1 步，有时能完成 2 步

报告:
  1/5: 完成 5 步中的 1 步序列
  2/5: 完成 5 步中的 2 步序列
  ...
  5/5: 完成所有 5 步序列

最终指标: Average length of completed sequence
```

### 关键结果参考

| 方法 | CALVIN Avg Steps (0-5) |
|------|----------------------|
| HULC (SOTA 2021) | ~2.5 |
| MCIL | ~2.0 |
| RT-1 | ~1.8 |
| OpenVLA (zero-shot) | ~1.2 |
| π₀.₅ | ~3.0+ |

## RLBench — 大规模 100 任务基准

### 简介

RLBench (Robot Learning Benchmark) 是 Imperial College London 开发的**最大规模单任务操作基准**，100 个不同任务，适合验证模型的多任务学习能力。

### 任务示例

| 类别 | 任务数 | 示例 |
|------|--------|------|
| **抓取** | 15 | pickup_cup, pickup_pen |
| **堆叠** | 8 | stack_blocks, stack_wine |
| **插入** | 10 | peg_in_hole, insert_usb |
| **开关操作** | 12 | open_door, open_drawer, turn_tap |
| **工具使用** | 10 | sweep_to_dustpan, slide_block_to_target |
| **按钮/开关** | 8 | press_switch, push_button |
| **倒/舀** | 5 | scoop_food, pour_water |
| **其他** | 32 | phone_on_base, reach_and_drag, ... |

### 安装

```bash
# RLBench 需要 CoppeliaSim (原 V-REP)
# 下载 CoppeliaSim: https://www.coppeliarobotics.com/downloads
pip install rlbench
```

### 任务变体

```python
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode
from rlbench.tasks import OpenDoor, PickUpCup

env = Environment(ArmActionMode.ABS_JOINT_POSITION)
env.launch()

# 加载任务，支持多种变化
task = env.get_task(OpenDoor)
variations = task.variation_count()  # 每个任务有多个变体

for var_idx in range(variations):
    descriptions, obs = task.reset()
    # descriptions: 自然语言描述 (如 "open the door")
    # obs: 多视角图像 + 本体感知

    while True:
        action = policy.predict(obs, descriptions)
        obs, reward, terminate = task.step(action)
        if terminate:
            break
```

### RLBench 的独特优势

```
1. Waypoint 生成: 每个任务提供预定义的 critical waypoints
   → 可用于层级 VLA 的中间目标

2. Motion Planner: 内置路径规划器 (RRT)
   → 可直接从 waypoint 生成无碰撞轨迹
   → 模拟"理想 low-level controller"

3. 语言描述的泛化: 每个任务有多语言描述变体
   - "open the door"
   - "pull the door handle to open it"
   - "open the door by pulling the handle towards you"
   → 测试 VLA 的语言鲁棒性
```

## CALVIN vs RLBench vs LIBERO

| | CALVIN | RLBench | LIBERO |
|---|--------|---------|--------|
| **任务数** | 34 子任务组合 | 100 | 130 |
| **核心指标** | 连续完成步数 | 单任务成功率 | 泛化成功率 |
| **语言指令** | ✅ 多步指令链 | ✅ 变体描述 | ✅ 单句指令 |
| **泛化测试** | 跨环境 (D→A) | 多任务 | 4 维度 (Object/Goal/...) |
| **物理引擎** | PyBullet | CoppeliaSim | MuJoCo/robosuite |
| **社区热度** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **VLA 首选** | 语言条件 | 大规模多任务 | 最新 VLA 标准 |

## 自检问题

### 基础关
- [ ] 我理解 CALVIN 的核心挑战（多步语言指令 + 跨环境泛化）
- [ ] 我知道 RLBench 的 100 任务覆盖了哪些类型
- [ ] 我能对比 CALVIN / RLBench / LIBERO 的适用场景

### 进阶关
- [ ] 我理解 CALVIN 的 "平均完成序列长度" 指标
- [ ] 我了解 RLBench 的 waypoint + motion planner 设计
- [ ] 我知道如何用 waypoint 做层级 VLA 的中层规划

### 实战关
- [ ] 我在 CALVIN 上评估过 VLA 模型
- [ ] 我在 RLBench 上跑过多任务 benchmark
- [ ] 我对比过 VLA 在三个基准上的泛化模式

## 关联笔记
- [[仿真与工具索引]] — 回到工具索引
- [[robosuite与模仿学习框架]] — LIBERO benchmark 详解
- [[PyBullet与Gymnasium]] — CALVIN 的物理后端
- [[../../02-VLA/VLA模型总览]] — VLA 在 CALVIN/RLBench 上的性能
