# MuJoCo

> DeepMind 维护的高性能物理引擎——机器人学习的标准仿真环境。MuJoCo (Multi-Joint dynamics with Contact) 以**快速隐式积分**和**精确接触力学**著称，是 VLA 训练和评估的首选物理后端。

## 为什么 MuJoCo 是机器人学习的首选？

| 特性 | MuJoCo | PyBullet | Isaac Sim | Gazebo |
|------|--------|----------|-----------|--------|
| **仿真速度** | ⚡ 极快 (隐式积分) | ⚡ 快 | ⚡ 中等 (GPU 并行) | ⚡ 慢 |
| **接触力学** | 精确 | 近似 | 中等 | 中等 |
| **GPU 并行** | 有限 | 无 | ✅ 大规模 | ❌ |
| **安装难度** | 一行 pip | 一行 pip | 复杂 (需 NVIDIA) | 中等 |
| **VLA 集成** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| **社区** | 最大 (RL/VLA) | 大 | 快速增长 | 大 (ROS) |

## 安装与基础

```bash
pip install mujoco
pip install gymnasium[mujoco]  # 标准 RL 接口
pip install dm_control          # DeepMind 控制套件 (可选)
```

### 第一个 MuJoCo 程序

```python
import mujoco
import numpy as np
import time

# 1. 加载模型
xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>
    <geom type="plane" size="1 1 0.1" rgba="0.5 0.5 0.5 1"/>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 2. 仿真循环
for step in range(1000):
    # 施加力（向下的重力 + 随机力）
    data.ctrl = np.zeros(model.nu)
    data.qfrc_applied[2] = -9.81 * 0.01  # 重力

    mujoco.mj_step(model, data)  # 前进一步

    # 读取状态
    pos = data.qpos.copy()  # [x, y, z, qw, qx, qy, qz]
    vel = data.qvel.copy()

    if step % 100 == 0:
        print(f"Step {step}: position={pos[:3]}")
```

## MJCF (MuJoCo XML Format) 深度解析

### 完整的机械臂定义

```xml
<mujoco model="franka_panda">
  <compiler angle="radian" meshdir="meshes"/>

  <!-- 材质 -->
  <asset>
    <texture name="metal" type="2d" file="metal.png"/>
    <material name="link_mat" texture="metal" rgba="1 1 1 1"/>
    <mesh name="link0" file="link0.stl"/>
    <mesh name="link1" file="link1.stl"/>
  </asset>

  <!-- 默认设置 -->
  <default>
    <joint limited="true" damping="0.1" armature="0.01"/>
    <geom rgba="0.6 0.6 0.6 1" solref="0.02 1"/>
    <motor ctrlrange="-1 1"/>
  </default>

  <worldbody>
    <!-- 基座 -->
    <body name="base_link" pos="0 0 0">
      <geom name="base_geom" type="mesh" mesh="link0" material="link_mat"/>
    </body>

    <!-- 第 1 关节 -->
    <body name="link1" pos="0 0 0.333">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
      <geom name="link1_geom" type="mesh" mesh="link1" material="link_mat"/>
    </body>

    <!-- 末端执行器 -->
    <body name="end_effector" pos="0 0 0.1">
      <geom name="ee_geom" type="sphere" size="0.02" rgba="1 0 0 1"/>
      <!-- 力传感器 -->
      <site name="ee_site" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>

  <!-- 执行器 -->
  <actuator>
    <motor name="joint1_motor" joint="joint1" gear="100" ctrlrange="-87 87"/>
    <motor name="joint2_motor" joint="joint2" gear="100" ctrlrange="-87 87"/>
    <!-- ... -->
  </actuator>

  <!-- 传感器 -->
  <sensor>
    <touch name="contact_sensor" site="ee_site"/>
    <force name="force_sensor" site="ee_site"/>
    <jointpos name="joint_pos"/>
    <jointvel name="joint_vel"/>
  </sensor>
</mujoco>
```

### 关键 XML 元素速查

| 元素 | 属性 | 含义 |
|------|------|------|
| `joint type="hinge"` | `axis, range` | 旋转关节 (1 DoF) |
| `joint type="slide"` | `axis, range` | 平移关节 |
| `joint type="ball"` | `range` (可选) | 球关节 (3 DoF) |
| `joint type="free"` | — | 6 DoF 自由体 |
| `geom type="box"` | `size="x y z"` | 矩形碰撞体 |
| `geom type="cylinder"` | `size="r h"` | 圆柱 |
| `geom type="capsule"` | `size="r h"` (fromto) | 胶囊体 (最快) |
| `geom type="mesh"` | `mesh="name"` | 三角网格 |
| `geom type="plane"` | — | 无限平面 (地面) |
| `motor` | `joint, gear, ctrlrange` | 关节力矩/位置控制 |
| `position` | `joint, kp, ctrlrange` | 位置伺服 |
| `site` | `pos, size` | 参考点 (传感器/相机安装点) |
| `camera` | `pos, fovy, resolution` | 渲染相机 |

### 碰撞体选择建议

```
简单形状 (box/cylinder/capsule/sphere):
  ✅ 碰撞检测极快
  ✅ 适合 RL (不需要视觉细节)
  ❌ 不精确 (大物体可能穿透)

Mesh (obj/stl):
  ✅ 精确外形
  ❌ 碰撞检测慢很多
  ❌ 非凸 mesh 不稳定

最佳实践:
  - 机器人连杆 = 简单几何体 (capsule/cylinder)
  - 操作物体 = 简单凸包近似
  - 只在需要精细接触时用 mesh
  - 地面/桌面 = plane
```

## Python API 深度

### 核心数据结构

```python
import mujoco

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# === 读取状态 ===
# 广义位置 (关节角 + 基座位姿)
qpos = data.qpos       # [nq,]
joint_angles = qpos[:7]  # 前 7 维 = 关节角

# 广义速度
qvel = data.qvel       # [nv,]

# 末端执行器位姿
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
ee_pos = data.xpos[ee_id]    # 世界坐标系位置 [x, y, z]
ee_mat = data.xmat[ee_id].reshape(3, 3)  # 世界坐标系旋转矩阵

# 雅可比矩阵
jac_pos = np.zeros((3, model.nv))  # 平移雅可比
jac_rot = np.zeros((3, model.nv))  # 旋转雅可比
mujoco.mj_jac(model, data, jac_pos, jac_rot, ee_pos, ee_id)

# === 设置控制 ===
# 位置控制 (需要 position actuator)
data.ctrl = target_joint_angles

# 力矩控制 (需要 motor actuator)
data.ctrl = joint_torques

# === 前向仿真 ===
mujoco.mj_step(model, data)  # 默认 dt = model.opt.timestep
# data.time += model.opt.timestep
# 更新: qpos, qvel, qacc, sensordata, ...

# === 手动前向运动学 (不前进仿真) ===
data.qpos[:7] = test_angles       # 设置关节角
mujoco.mj_forward(model, data)     # 只算 FK + 碰撞检测
# (data.qvel, data.qacc 不变)
```

### 力控与阻抗控制

```python
# 力矩控制 (直接设置关节力矩)
def torque_control(model, data, target_torques):
    data.ctrl = target_torques
    mujoco.mj_step(model, data)

# 阻抗控制 (模拟弹簧-阻尼)
def impedance_control(model, data, target_pos, kp=100, kd=10):
    """F = kp * (q_des - q) - kd * dq"""
    current_pos = data.qpos[:7]
    current_vel = data.qvel[:6]
    torques = kp * (target_pos - current_pos) - kd * current_vel
    data.ctrl = torques
    mujoco.mj_step(model, data)
```

### 传感器读取

```python
# 读取关节位置传感器
joint_pos_addr = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SENSOR, "joint_pos"
)
joint_pos = data.sensordata[joint_pos_addr:joint_pos_addr + 7]

# 读取力/力矩传感器
force_addr = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor"
)
force_3d = data.sensordata[force_addr:force_addr + 3]
torque_3d = data.sensordata[force_addr + 3:force_addr + 6]

# 读取接触力
for i in range(data.ncon):
    contact = data.contact[i]
    print(f"Contact: body {contact.geom1} ↔ {contact.geom2}")
    print(f"  Position: {contact.pos}")
    print(f"  Frame: {contact.frame.reshape(3,3)}")
```

### IK 验证

```python
# 用 MuJoCo 验证 IK 解的正确性
def verify_ik_with_mujoco(model, data, ik_joint_angles, expected_ee_pos):
    # 设置 IK 解
    data.qpos[:7] = ik_joint_angles

    # 前向运动学
    mujoco.mj_forward(model, data)

    # 读取末端位置
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
    actual_ee_pos = data.xpos[ee_id].copy()

    # 比较
    error = np.linalg.norm(actual_ee_pos - expected_ee_pos)
    if error > 1e-3:
        print(f"IK ERROR: {error:.6f}m")
    else:
        print(f"IK OK: error={error:.6f}m")
    return actual_ee_pos, error
```

## gymnasium 接口

### 标准 RL 环境

```python
import gymnasium as gym

# 内置环境
env = gym.make("HalfCheetah-v4", render_mode="human")
obs, info = env.reset(seed=42)

total_reward = 0
for step in range(1000):
    action = env.action_space.sample()  # 或策略输出
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
```

### 自定义 VLA 评估环境

```python
import gymnasium as gym
import mujoco
import numpy as np

class VLAEvaluationEnv(gym.Env):
    """封装 MuJoCo 为 VLA 评估的标准 gym 接口"""

    def __init__(self, xml_path, max_steps=200):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.step_count = 0

        # 定义动作空间 (增量末端位姿 Δx,Δy,Δz,Δr,Δp,Δy,gripper)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        # 定义观测空间 (图像 + 本体感知)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32),
            "language": gym.spaces.Text(256),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        # 随机初始化物体位置
        self._randomize_objects()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # 解析 VLA 输出 (Δ 末端位姿)
        delta_ee = action[:6]
        gripper = action[6] > 0.5  # 二值化夹爪

        # IK 解算 + 设置关节目标
        joint_target = self._ee_delta_to_joint(delta_ee)
        self.data.ctrl = joint_target

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # 计算 reward
        reward = self._compute_reward()
        terminated = self._check_success()
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return {
            "image": self._render_camera(),
            "state": np.concatenate([self.data.qpos[:7], self.data.qvel[:7]]),
            "language": self.task_instruction,
        }
```

## 渲染

### 离线渲染 (CPU)

```python
import mujoco

# 修改渲染配置
model.vis.global_.offwidth = 640
model.vis.global_.offheight = 480

# 创建渲染器
with mujoco.Renderer(model, 640, 480) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="front_camera")
    pixels = renderer.render()  # (480, 640, 3) uint8
```

### 多相机渲染

```xml
<!-- MJCF 中定义多个相机 -->
<camera name="front_cam" pos="1.5 0 1.2" xyaxes="0 -1 0 0.4 0 0.9"/>
<camera name="wrist_cam" pos="0 0 0.05" xyaxes="1 0 0 0 1 0" mode="body" body="end_effector"/>
<camera name="top_cam" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
```

```python
# 从不同相机渲染
for cam_name in ["front_cam", "wrist_cam", "top_cam"]:
    renderer.update_scene(data, camera=cam_name)
    pixels = renderer.render()
    # 保存或显示
```

## 与 VLA 的完整集成

### VLA 策略评估循环

```python
def evaluate_vla_in_mujoco(
    vla_model,        # VLA 模型 (π_θ: obs → action)
    env_xml_path,     # MuJoCo 场景
    task_instruction, # 语言指令
    num_episodes=100,
    max_steps=200,
):
    model = mujoco.MjModel.from_xml_path(env_xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 640, 480)

    successes = 0
    for ep in range(num_episodes):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        obs_history = []  # 累积观测

        for step in range(max_steps):
            # 渲染当前视图
            renderer.update_scene(data, camera="front_cam")
            image = renderer.render()

            # VLA 推理
            obs = {
                "image": image,
                "proprio": data.qpos[:7],
                "language": task_instruction,
            }
            action = vla_model.predict(obs)

            # 执行动作 (IK → 关节目标 → MuJoCo)
            joint_target = solve_ik(model, data, action)
            data.ctrl = joint_target
            mujoco.mj_step(model, data)

            # 检查成功
            if check_task_success(model, data):
                successes += 1
                break

    success_rate = successes / num_episodes
    print(f"VLA Success Rate: {success_rate:.2%}")
    return success_rate
```

### 数据生成 (仿真 → 训练数据)

```python
def generate_simulation_data(
    policy,       # 专家策略 (或脚本策略)
    env_xml,      # 场景
    num_episodes=1000,
):
    """在 MuJoCo 中用脚本策略自动收集训练数据"""
    episodes = []
    for ep in range(num_episodes):
        steps = []
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        while not done:
            image = render()
            state = data.qpos.copy()
            action = policy(image, state)
            data.ctrl = action
            mujoco.mj_step(model, data)

            steps.append({
                "observation": {"image": image, "state": state},
                "action": action,
                "reward": compute_reward(),
            })

        episodes.append({"steps": steps, "success": done})
    return episodes
```

## 高级技巧

### 1. 并行仿真 (多进程)

```python
from multiprocessing import Pool

def run_one_episode(args):
    seed, vla_weights = args
    model = mujoco.MjModel.from_xml_path("robot.xml")
    data = mujoco.MjData(model)
    # 用 seed 随机化初始状态
    np.random.seed(seed)
    # ... 运行 episode
    return success

with Pool(8) as pool:
    results = pool.map(run_one_episode, [(i, weights) for i in range(64)])
```

### 2. 域随机化

```python
def randomize_physics(model):
    """随机化物理参数 → 增强 sim-to-real 转移"""
    model.opt.timestep *= np.random.uniform(0.9, 1.1)
    model.dof_damping[:] *= np.random.uniform(0.8, 1.2, model.nv)
    model.dof_armature[:] *= np.random.uniform(0.9, 1.1, model.nv)
    model.body_mass[:] *= np.random.uniform(0.95, 1.05, model.nbody)
    model.geom_friction[:] *= np.random.uniform(0.8, 1.2, (model.ngeom, 3))
    # 不要随机化 actuator gain → 可能导致不稳定
```

### 3. 碰撞检测调试

```python
# 检查两个 body 是否碰撞
def check_collision(model, data, body1_name, body2_name):
    id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
    id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_body = model.geom_bodyid[contact.geom1]
        geom2_body = model.geom_bodyid[contact.geom2]
        if (geom1_body == id1 and geom2_body == id2) or \
           (geom1_body == id2 and geom2_body == id1):
            return True
    return False
```

## 自检问题

### 基础关
- [ ] 我能读懂和编写 MJCF XML 文件
- [ ] 我知道如何加载模型、设置关节角、步进仿真
- [ ] 我能用 gymnasium 接口运行标准环境
- [ ] 我理解 qpos/qvel/qacc/ctrl 的含义

### 进阶关
- [ ] 我能用 Python API 实现 IK 验证
- [ ] 我能渲染多相机图像
- [ ] 我理解接触力学和碰撞体选择
- [ ] 我能写自定义 gymnasium 环境用于 VLA 评估

### 实战关
- [ ] 我评估过 VLA 模型在 MuJoCo 中的成功率
- [ ] 我用 MuJoCo 自动生成过训练数据
- [ ] 我实现过域随机化 sim-to-real pipeline
- [ ] 我在 MuJoCo 中搭建过自定义机器人仿真场景

## 关联笔记
- [[Blender与3D建模]] — 创建 mesh 导入 MuJoCo
- [[PyBullet与Gymnasium]] — 轻量替代方案
- [[Isaac Sim与Isaac Lab]] — GPU 加速替代
- [[rosbosuite与模仿学习框架]] — 基于 MuJoCo 的 benchmark
- [[../../01-基础理论/机器人运动学]] — FK/IK 理论基础
- [[../../01-基础理论/强化学习]] — MuJoCo 中的 RL 训练
- [[../../02-VLA/行为克隆]] — 在 MuJoCo 中评估 BC 策略
- [[仿真与工具索引]] — 回到工具索引
