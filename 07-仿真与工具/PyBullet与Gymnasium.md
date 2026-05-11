# PyBullet 与 Gymnasium

> 最易上手的物理仿真器——一行 pip install 就能在机器人仿真中跑 RL。

## 定位

PyBullet 是 Bullet Physics 的 Python 绑定，长期是机器人学习社区的主流快速原型工具。尽管 MuJoCo 现在免费后使用更广，PyBullet 仍然在教育和快速原型场景中占据重要地位。

```
PyBullet 的定位: "快速跑通一个 idea，不用折腾配置"
MuJoCo 的定位:   "工业级 RL 和 VLA 训练"
Isaac Sim 的定位: "视觉策略 + 大规模 GPU 并行"
```

## 为什么还在用 PyBullet？

| 优势 | 劣势 |
|------|------|
| ✅ 一行 pip 安装 | ❌ 接触力学不如 MuJoCo 精确 |
| ✅ URDF/SDF 原生支持 | ❌ 渲染质量一般 |
| ✅ 丰富的文档和社区 | ❌ GPU 并行能力有限 |
| ✅ 跨平台 (Win/Mac/Linux) | ❌ 复杂场景不稳定 |
| ✅ pybullet_envs (标准环境) | ❌ 维护不如 MuJoCo 活跃 |
| ✅ VR/远程可视化 | |

## 安装

```bash
pip install pybullet
pip install gymnasium  # 标准 RL 接口
pip install stable-baselines3  # RL 算法 (可选)
```

## 基础 API

### Hello World — 加载机器人

```python
import pybullet as p
import pybullet_data
import time

# 连接 (GUI 或 DIRECT 无头)
p.connect(p.GUI)  # p.GUI 可视化, p.DIRECT 无头

# 加载资源
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 加载地面
plane_id = p.loadURDF("plane.urdf")

# 加载 Franka Panda 机器人
robot_id = p.loadURDF("franka_panda/panda.urdf",
                       basePosition=[0, 0, 0],
                       useFixedBase=True)

# 获取关节信息
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    print(f"Joint {i}: {info[1].decode('utf-8')} type={info[2]}")

# 仿真循环
for step in range(10000):
    p.stepSimulation()
    time.sleep(1/240.)  # 实时

p.disconnect()
```

### 运动控制

```python
# 方法 1: 位置控制
for joint_idx in range(7):
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_idx,
        controlMode=p.POSITION_CONTROL,
        targetPosition=target_angle,
        targetVelocity=0,
        force=100,            # 最大力矩
        maxVelocity=1.0,
    )

# 方法 2: 速度控制
p.setJointMotorControl2(
    bodyUniqueId=robot_id,
    jointIndex=joint_idx,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity=1.0,
    force=100,
)

# 方法 3: 力矩控制 (最底层)
p.setJointMotorControl2(
    bodyUniqueId=robot_id,
    jointIndex=joint_idx,
    controlMode=p.TORQUE_CONTROL,
    force=torque_value,
)
```

### 读取状态

```python
# 关节状态
joint_state = p.getJointState(robot_id, joint_idx)
joint_pos = joint_state[0]  # 角度 (rad)
joint_vel = joint_state[1]  # 角速度

# 末端执行器位姿 (需要知道 link index)
ee_link_idx = 11  # Franka 的第 11 号 link
ee_state = p.getLinkState(robot_id, ee_link_idx)
ee_pos = ee_state[0]       # [x, y, z]
ee_orn = ee_state[1]       # 四元数 [x, y, z, w]

# 逆运动学 (内置!)
ee_target = [0.5, 0.0, 0.3]
joint_poses = p.calculateInverseKinematics(
    robot_id, ee_link_idx, ee_target,
    # 可选: 指定末端朝向
    # targetOrientation=p.getQuaternionFromEuler([0, np.pi, 0])
)
```

### 相机渲染

```python
# 设置视角
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.5],
)

# 获取渲染图像 (RGBA)
width, height = 640, 480
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[1, 1, 1],
    cameraTargetPosition=[0, 0, 0.5],
    cameraUpVector=[0, 0, 1],
)
proj_matrix = p.computeProjectionMatrixFOV(
    fov=60, aspect=width/height,
    nearVal=0.1, farVal=10.0,
)
img = p.getCameraImage(width, height,
                        viewMatrix=view_matrix,
                        projectionMatrix=proj_matrix)
# img = [w, h, rgba, depth, segmentation_mask]
rgb = img[2]      # (h, w, 4) RGBA
depth = img[3]    # (h, w) depth buffer
seg = img[4]      # (h, w) object mask
```

## Gymnasium 标准接口

```python
import gymnasium as gym

# PyBullet 标准环境 (需要安装 pybullet_envs)
# pip install pybullet-envs (社区维护)
env = gym.make("HalfCheetahBulletEnv-v0")

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### 自定义 Gymnasium + PyBullet 环境

```python
import gymnasium as gym
import pybullet as p
import numpy as np

class PyBulletReachEnv(gym.Env):
    """PyBullet VLA 评估环境: 末端到达任务"""

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = gym.spaces.Box(
            low=-0.05, high=0.05, shape=(3,), dtype=np.float32  # Δx,Δy,Δz
        )
        self.observation_space = gym.spaces.Dict({
            "ee_pos": gym.spaces.Box(-1, 1, (3,), np.float32),
            "target_pos": gym.spaces.Box(-1, 1, (3,), np.float32),
            "image": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8),
        })

        if render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        self.robot = p.loadURDF("franka_panda/panda.urdf",
                                 basePosition=[0, 0, 0], useFixedBase=True)
        self.ee_idx = 11
        self.target_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 重置机器人到初始姿态
        for i in range(7):
            p.resetJointState(self.robot, i, 0)
        # 随机目标
        self.target_pos = self.np_random.uniform(
            low=[0.3, -0.3, 0.1], high=[0.7, 0.3, 0.5]
        )
        return self._get_obs(), {}

    def step(self, action):
        # action = Δ 末端位置
        current_ee = np.array(p.getLinkState(self.robot, self.ee_idx)[0])
        target_ee = current_ee + action

        # IK 解算
        joint_angles = p.calculateInverseKinematics(
            self.robot, self.ee_idx, target_ee
        )

        # 位置控制
        for i in range(7):
            p.setJointMotorControl2(
                self.robot, i, p.POSITION_CONTROL,
                targetPosition=joint_angles[i], force=200
            )

        p.stepSimulation()

        # Reward: 负距离
        ee_pos = np.array(p.getLinkState(self.robot, self.ee_idx)[0])
        dist = np.linalg.norm(ee_pos - self.target_pos)
        reward = -dist
        terminated = dist < 0.02
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        ee_pos = p.getLinkState(self.robot, self.ee_idx)[0]
        # 简化渲染 (小尺寸, 灰度图)
        img = self._render_camera()
        return {
            "ee_pos": np.array(ee_pos, dtype=np.float32),
            "target_pos": self.target_pos,
            "image": img,
        }
```

## PyBullet vs MuJoCo 迁移指南

```python
# ─── MuJoCo ───              ─── PyBullet ───
model.nq                       p.getNumJoints(robot)
data.qpos                      p.getJointState(robot, i)
mujoco.mj_step(model, data)    p.stepSimulation()
data.ctrl = target             p.setJointMotorControl2(...)
mj_name2id(model, ...)         joint_name_to_id dict (手动)
mujoco.mj_forward(model, data) p.stepSimulation()  # 已含 FK
model.opt.timestep             p.setTimeStep(dt)
```

## 自检问题

### 基础关
- [ ] 我能用 PyBullet 加载 URDF 机器人
- [ ] 我理解三种控制模式（位置/速度/力矩）的区别
- [ ] 我能用内置 IK 解算末端目标位置

### 进阶关
- [ ] 我能写自定义 gymnasium + PyBullet 环境
- [ ] 我理解 PyBullet 和 MuJoCo 的 API 对应关系
- [ ] 我能从 PyBullet 渲染图像和多视角分割 mask

### 实战关
- [ ] 我在 PyBullet 中训练过 RL 策略
- [ ] 我做过 PyBullet → MuJoCo 的代码迁移
- [ ] 我在 PyBullet 中评估过 VLA 策略

## 关联笔记
- [[仿真与工具索引]] — 回到工具索引
- [[MuJoCo]] — 更精确的工业级替代
- [[Isaac Sim与Isaac Lab]] — 高质量渲染替代
- [[../../01-基础理论/机器人运动学]] — IK/FK 理论基础
- [[../../01-基础理论/强化学习]] — PyBullet 中的 RL 训练
