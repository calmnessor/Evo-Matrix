# MuJoCo

> DeepMind 维护的高性能物理引擎——机器人学习的标准仿真环境

## 安装

```bash
pip install mujoco
pip install gymnasium[mujoco]
```

## 核心概念

### MJCF (MuJoCo XML Format)

描述机器人/场景的标准 XML 格式：

```xml
<mujoco model="robot">
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-2.5 2.5"/>
      <geom name="link1" type="capsule" fromto="0 0 0 0.5 0 0" size="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="100"/>
  </actuator>
</mujoco>
```

### 关键 MJCF 字段

| 字段 | 含义 |
|------|------|
| `joint type="hinge"` | 旋转关节 |
| `joint type="slide"` | 平移关节 |
| `joint axis="0 0 1"` | 旋转轴 |
| `joint range` | 关节限位 (rad) |
| `motor gear` | 力矩放大 |
| `geom type="capsule"` | 胶囊体碰撞 |
| `geom rgba` | 颜色 RGBA |

## Python API

```python
import mujoco
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("robot_arm.xml")
data = mujoco.MjData(model)

# 设置关节角度
data.qpos[0] = 0.5  # joint 1
data.qpos[1] = -0.3 # joint 2

# 自动算正运动学
mujoco.mj_forward(model, data)

# 读取末端执行器位置
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "end_effector")
ee_pos = data.geom_xpos[ee_id]
```

## gymnasium 接口

```python
import gymnasium as gym

env = gym.make("HalfCheetah-v4")
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## 在 VLA 中的应用

1. **策略评估**: 把 VLA 输出的动作给 MuJoCo → 看任务成功率
2. **数据生成**: 在仿真中自动收集训练数据
3. **RL 训练**: MuJoCo + PPO 训练策略
4. **IK 验证**: 算出的 IK 用 MuJoCo 的 FK 验证

## 自检问题
- [ ] 我能读懂 MJCF XML 文件
- [ ] 我知道如何加载自定义模型并设置关节角度
- [ ] 我能用 gymnasium 接口训练/评估策略

## 关联笔记
- [[机器人运动学]] — 在 MuJoCo 中验证 FK/IK
- [[强化学习]] — MuJoCo 是标准 RL 训练环境
- [[VLA模型总览]]
- [[03-仿真与工具/仿真与工具索引|仿真工具索引]]
