# 🤖 Day 6: 具身智能概览与机器人基础（8 小时）

> **口号**: "理解机器人，才能理解 VLA 要解决什么问题！"  
> **目标**: 建立具身智能领域全景图，理解机器人基础概念和主流学习范式  
> **为什么重要**: 不了解"动作空间"就不知道 VLA 输出什么，不了解"数据集"就不知道 VLA 训练什么

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | 具身智能全景 | 1h | [ ] |
| 2 | 机器人基础 + 运动学（⭐核心） | 3h | [ ] |
| 3 | 机器人学习三大范式 | 1h | [ ] |
| 4 | Open X-Embodiment 数据集 | 1h | [ ] |
| 5 | 仿真环境 + MuJoCo 入门 | 1h | [ ] |
| 6 | 费曼挑战 | 1h | [ ] |

---

## 模块 1: 具身智能全景（1.5 小时）

### 1.1 什么是具身智能？ — 30min

```
传统 AI (Internet AI):    具身智能 (Embodied AI):
  输入: 文本/图像              输入: 多传感器（视觉、触觉、力觉）
  输出: 文本/标签              输出: 物理动作（关节角度、速度）
  环境: 数字世界              环境: 物理世界
  反馈: 标签/评分              反馈: 物理规律（碰到了就是碰到了）
```

```
具身智能的核心挑战:

  感知 (Perception) ──→ 决策 (Decision) ──→ 控制 (Control)
       │                      │                     │
  "我看到什么？"        "我该做什么？"       "我怎么做？"
  摄像头、深度           策略网络              运动规划
  触觉传感器             VLA 模型              阻抗控制
       │                      │                     │
       └──────────────────────┴─────────────────────┘
                             │
                    从像素到动作的端到端映射
                            ↖ VLA 做的事情 ↗
```

### 1.2 领域发展时间线 — 20min

```
2015-2018: 深度强化学习在机器人中的应用
  DQN, DDPG, PPO 控制简单任务（倒立摆、抓取）

2019-2021: 模仿学习与行为克隆
  BC-Z, 使用演示数据训练

2022: RT-1 — Robotics Transformer
  首次将 Transformer 大规模应用于机器人

2023: RT-2 — Vision-Language-Action
  首次用 VLM 直接输出动作 token
  ACT, Diffusion Policy 发布

2024: 通用 VLA 元年
  Octo, OpenVLA, π₀ (PI-Zero), GR00T
  开源 + 大规模 + 多任务

2025: 部署与 scaling
  更大规模、更通用、sim-to-real
```

### 1.3 核心研究机构与团队 — 20min

| 机构 | 代表作 | 特点 |
|------|--------|------|
| Google DeepMind | RT-1, RT-2, RT-X | 大规模、端到端 |
| Stanford IRIS | ACT, Diffusion Policy | 创新的动作表征 |
| UC Berkeley | Octo, SERL | 开源通用策略 |
| CMU | In-the-Wild Human Imitation | 人形机器人 |
| NVIDIA | GR00T | 工业级基础模型 |
| Physical Intelligence | π₀ | 大规模通用操作 |
| 清华/北大/上交 | 各种 VLA 相关工作 | 开源贡献 |

### 1.4 VLA 的三大技术路线 — 10min

```
路线 1: LLM + 动作 Token 化 (RT-2 风格)
  大语言模型 → 把动作编码成 token → 自回归生成

路线 2: 扩散策略 (Diffusion Policy 风格)
  从噪声中逐步去噪 → 生成平滑的动作序列

路线 3: 条件 VAE 生成 (ACT 风格)
  编码器-解码器 → 从观测生成动作序列
```

---

## 模块 2: 机器人运动学与基础（3 小时）

> ⭐ 运动学是理解"VLA 输出动作 → 机器人末端怎么动"的核心桥梁

### 2.1 机械臂基础 — 30min

```
自由度 (DOF - Degree of Freedom):
  每个可独立控制的关节 = 1 个 DOF
  典型 6-DOF 机械臂: 6 个旋转关节 → 任意位置 + 姿态
  常见 7-DOF 机械臂: 增加冗余度（避开障碍更灵活）

末端执行器 (End-Effector):
  夹爪 (Gripper): 平行夹爪、三指夹爪
  吸盘 (Suction Cup): 真空吸附

动作空间的三种表示:
  1. 关节空间 (Joint Space): [θ₁, θ₂, ..., θ₆] ← VLA 最常用
  2. 笛卡尔空间 (Task Space): [x, y, z, roll, pitch, yaw]
  3. 增量控制: [Δθ₁, ..., Δθ₆] 或 [Δx, Δy, Δz, Δr, Δp, Δy]
```

### 2.2 坐标系 — 20min

```
世界坐标系 (World Frame):
  固定的参考系，通常在地面

基座坐标系 (Base Frame):
  机械臂底座

工具坐标系 (Tool/EE Frame):
  末端执行器

相机坐标系 (Camera Frame):
  眼在手 (Eye-in-Hand): 相机在机械臂上
  眼在外 (Eye-to-Hand): 相机在外部固定位置

VLA 中一般用末端执行器的姿态来控制:
  action = [dx, dy, dz, droll, dpitch, dyaw, gripper_open]
```

### 2.3 正运动学 (Forward Kinematics, FK) — 40min

> **核心问题**: 已知每个关节的角度，求末端执行器在空间中的位置和姿态。

```
正运动学 = 关节空间 → 笛卡尔空间

输入: [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]  (6 个关节角度)
输出: [x, y, z, roll, pitch, yaw]   (末端位置 + 姿态)
```

**简单例子 — 2 连杆平面臂**:

```
         y ↑
           │     末端 (x, y)
           │     ●
           │    /   <- 连杆 2, 长度 L₂
           │   / θ₂
           │  ●     <- 关节 2
           │ /  <- 连杆 1, 长度 L₁
           │/ θ₁
    ───────●────────→ x
         基座

正运动学（一步算出末端位置）:
  x = L₁·cos(θ₁) + L₂·cos(θ₁ + θ₂)
  y = L₁·sin(θ₁) + L₂·sin(θ₁ + θ₂)

这就是 FK! 给定 θ₁, θ₂ → 直接算出末端在哪
```

**3D 空间中的 FK — 齐次变换矩阵**:

```python
import numpy as np

def rot_x(angle):
    """绕 X 轴旋转的 4×4 齐次变换矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1],
    ])

def rot_z(angle):
    """绕 Z 轴旋转的 4×4 齐次变换矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])

def translate(dx, dy, dz):
    """平移的 4×4 齐次变换矩阵"""
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1],
    ])

# FK: 将所有关节的变换矩阵连乘
# T_total = T₁ · T₂ · T₃ · ... · T₆
# 末端位姿 = T_total @ [0, 0, 0, 1]^T

def forward_kinematics_3d(joint_angles, link_lengths):
    """
    joint_angles: [θ₁, θ₂, θ₃, ...]  每个关节的旋转角
    link_lengths: [L₁, L₂, L₃, ...]   每段连杆的长度
    """
    T = np.eye(4)  # 从单位矩阵开始
    for i, (theta, L) in enumerate(zip(joint_angles, link_lengths)):
        # 绕 Z 旋转（典型旋转关节）
        T = T @ rot_z(theta)
        # 沿 X 平移（下一段连杆）
        T = T @ translate(L, 0, 0)
    # 末端位置 = T 的平移分量
    position = T[:3, 3]  # [x, y, z]
    return position, T

# 示例: 3 连杆臂
angles = [np.pi/4, np.pi/6, -np.pi/6]  # 45°, 30°, -30°
links = [1.0, 0.8, 0.5]
pos, T = forward_kinematics_3d(angles, links)
print(f"末端位置: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
```

**自检题**: 如果所有关节角度都是 0，末端位置在哪里？验证你的答案。

---

### 2.4 逆运动学 (Inverse Kinematics, IK) — 40min

> **核心问题**: 已知想要末端到达的位置和姿态，求各个关节的角度。

```
逆运动学 = 笛卡尔空间 → 关节空间

输入: [x, y, z, roll, pitch, yaw]   (末端目标位姿)
输出: [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]    (6 个关节角度)
```

**IK 比 FK 难得多！** 原因:
```
1. 非线性方程组 → 可能没有解析解
2. 一个末端位姿 → 可能对应多组关节解（"肘向上"vs"肘向下"）
3. 可能无解（目标超出工作空间）
```

**2 连杆臂的 IK（解析解）**:

```python
def inverse_kinematics_2link(x, y, L1, L2):
    """
    2 连杆平面臂的 IK 解析解
    返回: (θ₁, θ₂) 两个可能的解
    """
    # 余弦定理
    cos_θ2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_θ2 = np.clip(cos_θ2, -1, 1)  # 防止数值误差

    θ2 = np.arccos(cos_θ2)  # 有两个解: ±θ2

    θ1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(θ2), L1 + L2 * np.cos(θ2))

    return θ1, θ2

# 测试
θ1, θ2 = inverse_kinematics_2link(1.2, 0.8, L1=1.0, L2=0.8)
print(f"IK 解: θ₁={np.degrees(θ1):.1f}°, θ₂={np.degrees(θ2):.1f}°")

# 验证（用 FK 算回去）
x_check = 1.0 * np.cos(θ1) + 0.8 * np.cos(θ1 + θ2)
y_check = 1.0 * np.sin(θ1) + 0.8 * np.sin(θ1 + θ2)
print(f"验证 FK: x={x_check:.3f}, y={y_check:.3f}")  # 应该 ≈ (1.2, 0.8)
```

**6-DOF 机械臂 IK 的数值解法**:

```python
def inverse_kinematics_numerical(target_pos, initial_angles, link_lengths,
                                  max_iter=100, lr=0.01):
    """
    用梯度下降数值求解 IK
    适用于没有解析解的复杂机械臂
    """
    angles = np.array(initial_angles, dtype=float)

    for i in range(max_iter):
        # 1. 用当前角度算 FK
        current_pos, _ = forward_kinematics_3d(angles, link_lengths)

        # 2. 计算位置误差
        error = target_pos - current_pos
        if np.linalg.norm(error) < 1e-6:
            print(f"IK 收敛，迭代 {i+1} 次")
            break

        # 3. 用数值微分估算雅可比 → 更新角度
        jacobian = numerical_jacobian(angles, link_lengths)
        delta_angles = lr * jacobian.T @ error
        angles += delta_angles

    return angles

def numerical_jacobian(angles, link_lengths, eps=1e-6):
    """用有限差分估算雅可比矩阵"""
    pos, _ = forward_kinematics_3d(angles, link_lengths)
    jac = np.zeros((3, len(angles)))
    for i in range(len(angles)):
        angles_perturbed = angles.copy()
        angles_perturbed[i] += eps
        pos_perturbed, _ = forward_kinematics_3d(angles_perturbed, link_lengths)
        jac[:, i] = (pos_perturbed - pos) / eps
    return jac
```

**VLA 与 IK 的关系**:
```
VLA 可以直接输出两种控制:
  1. 笛卡尔空间动作: [dx, dy, dz, dr, dp, dy]
     → 需要 IK 转换为关节指令
     → OpenVLA、RT-2 大多采用这种方式

  2. 关节空间动作: [Δθ₁, ..., Δθ₆]
     → 不需要 IK，直接发给电机
     → 但 VLA 需要学习关节空间的控制
```

**自检题**: 为什么大多数 VLA 输出笛卡尔空间的动作而非关节空间？

---

### 2.5 DH 参数法 (Denavit-Hartenberg) — 30min

> **标准化的运动学建模方法** — 用 4 个参数描述一个关节。

```
DH 参数 = 机器人运动学的"通用语言"

每个关节用 4 个 DH 参数描述:
  a (link length):   连杆长度 — 两个相邻 Z 轴之间的公共法线长度
  α (link twist):    连杆扭转 — 绕 X 轴从一个 Z 轴到下一个 Z 轴的角度
  d (link offset):   关节偏移 — 沿 Z 轴从一个 X 轴到下一个 X 轴的距离
  θ (joint angle):   关节角度 — 绕 Z 轴从一个 X 轴到下一个 X 轴的角度

对于旋转关节: θ 是变量，其余是常数
对于平移关节: d 是变量，其余是常数
```

**DH 变换矩阵（标准 DH）**:

```python
def dh_transform(a, alpha, d, theta):
    """
    DH 参数 → 4×4 齐次变换矩阵
    标准 DH 约定: T = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1],
    ])

# 示例: Franka Panda 机械臂的 DH 参数（简化版）
panda_dh_params = [
    # (a, alpha, d, theta) — theta 是变量
    (0,      -np.pi/2, 0.333, 0),   # 关节 1
    (0,       np.pi/2, 0,      0),   # 关节 2
    (0.0825,  np.pi/2, 0.316, 0),   # 关节 3
    (-0.0825,-np.pi/2, 0,      0),   # 关节 4
    (0,       np.pi/2, 0.384, 0),   # 关节 5
    (0.088,   np.pi/2, 0,      0),   # 关节 6
    (0,       0,       0.107,  0),   # 关节 7 (末端)
]

def panda_fk(joint_angles):
    """用 DH 参数算 Panda 机械臂的正运动学"""
    T = np.eye(4)
    for (a, alpha, d, _), theta in zip(panda_dh_params, joint_angles):
        T = T @ dh_transform(a, alpha, d, theta)
    return T[:3, 3], T  # 位置 + 完整位姿

# 测试
q = [0.1, -0.3, 0.2, -1.0, 0.5, 0.3, 0.0]
pos, T = panda_fk(q)
print(f"Panda 末端位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
```

**为什么 VLA 工程师要懂 DH 参数？**

```
1. 理解 VLA 训练数据中 action 的实际含义
   →"dx"是相对于哪个坐标系的？

2. 在新机器人上部署 VLA 时
   → 需要知道该机器人的 DH 参数来设置控制接口

3. 调试 IK 和控制问题时
   → DH 参数错了，IK 就会算错
```

---

### 2.6 雅可比矩阵与奇异性 — 30min

> **雅可比矩阵 = 关节速度 → 末端速度的映射**

```
雅可比矩阵 J 的定义:

ẋ = J(θ) · θ̇

ẋ:  末端速度 [vx, vy, vz, ωx, ωy, ωz]  (6维)
θ̇: 关节速度 [θ̇₁, θ̇₂, ..., θ̇₆]      (6维)
J:  6×6 雅可比矩阵

J = [∂x/∂θ₁  ∂x/∂θ₂  ...  ∂x/∂θ₆]
    [∂y/∂θ₁  ∂y/∂θ₂  ...  ∂y/∂θ₆]
    [  ...     ...    ...    ...  ]
    [∂ωz/∂θ₁ ∂ωz/∂θ₂ ... ∂ωz/∂θ₆]
```

```python
def compute_jacobian_numerical(fk_func, joint_angles, eps=1e-6):
    """
    数值计算雅可比矩阵
    fk_func: 正运动学函数，返回 (position, rotation)
    """
    n = len(joint_angles)
    pos_ref, _ = fk_func(joint_angles)
    J = np.zeros((6, n))  # 6 DOF 末端 × n 个关节

    for i in range(n):
        q_plus = joint_angles.copy()
        q_plus[i] += eps
        pos_plus, _ = fk_func(q_plus)

        # 位置雅可比（后向差分）
        J[:3, i] = (pos_plus - pos_ref) / eps
        # 姿态雅可比（简化处理）
        # 完整版本需要从旋转矩阵提取角速度

    return J

# 使用雅可比实现更高效的 IK
def ik_with_jacobian(target_pos, initial_angles, link_lengths, max_iter=50):
    """使用雅可比伪逆的 IK 求解器"""
    angles = np.array(initial_angles, dtype=float)

    for i in range(max_iter):
        pos, _ = forward_kinematics_3d(angles, link_lengths)
        error = target_pos - pos

        if np.linalg.norm(error) < 1e-5:
            break

        J = compute_jacobian_numerical(
            lambda q: forward_kinematics_3d(q, link_lengths),
            angles
        )

        # 伪逆: J⁺ = J^T (J J^T)^(-1)
        # 或者直接用最小二乘: np.linalg.lstsq(J, error)
        delta_q, _, _, _ = np.linalg.lstsq(J[:3, :], error)
        angles += delta_q

    return angles
```

**奇异性 (Singularity)** — VLA 部署时最容易踩的坑:

```
奇异性 = 在某些关节配置下，雅可比矩阵秩亏，机械臂失去某些运动能力。

三类奇异性:
  1. 边界奇异性: 机械臂完全伸直或完全折叠
  2. 腕部奇异性: 关节 4 和关节 6 共线
  3. 内部奇异性: 特定关节组合导致

在奇异点附近:
  - 小的末端速度 → 巨大的关节速度 → 关节电机可能过载
  - IK 求解不稳定 → 数值解剧烈跳动
  - VLA 输出的笛卡尔动作可能无法执行

VLA 工程师怎么做:
  1. 限制 VLA 动作输出的范围（避免到达奇异配置）
  2. 在 IK 求解器中加入阻尼（Damped Least Squares）
  3. 动作空间选关节空间而非笛卡尔空间
```

```python
def detect_singularity(joint_angles, threshold=0.01):
    """检测当前关节配置是否接近奇异点"""
    J = compute_jacobian_numerical(
        lambda q: forward_kinematics_3d(q, [1.0]*len(joint_angles)),
        joint_angles
    )
    # 计算雅可比的最小奇异值
    _, S, _ = np.linalg.svd(J[:3, :])
    min_singular = S[-1]
    if min_singular < threshold:
        print(f"⚠️  接近奇异点！最小奇异值: {min_singular:.5f}")
        return True
    return False
```

---

### 2.7 常见传感器 — 15min

```
1. RGB 摄像头:
  最常用，VLA 主要输入源

2. 深度摄像头 (Depth):
  RealSense, Kinect → 提供 3D 信息

3. 关节编码器:
  提供当前关节角度（本体感知 proprioception）

4. 力/力矩传感器 (F/T Sensor):
  感知接触力，用于精细操作

5. 触觉传感器 (Tactile):
  GelSight 等，感知接触纹理
```

### 2.8 控制频率 — 10min

```
常见控制频率: 10-50 Hz
  - 10 Hz: 每 100ms 一个动作（VLA 推理一般这个频率）
  - 30 Hz: 每 33ms 一个动作（扩散策略可以做到）
  - 50 Hz+: 需要高速控制器（模型预测控制）

VLA 的挑战:
  大模型推理慢（100-500ms），而机器人需要实时响应
  → 需要 Action Chunking: 一次预测未来 N 步动作
```

---

## 模块 3: 机器人学习三大范式（1 小时）

### 3.1 模仿学习 (Imitation Learning, IL) — 30min

```
核心思想: 从人类演示中学习

数据收集:
  1. 遥操作 (Teleoperation): 人远程控制机械臂
  2. 动觉示教 (Kinesthetic Teaching): 人手拖动机器人
  3. 手持夹爪示教: 用简易装置采集数据

行为克隆 (Behavior Cloning, BC):
  给定 (观察, 动作) 对 → 监督学习 → 预测动作

优点: 简单、高效、不需要 reward 函数
缺点: 分布漂移 → 误差会累积 → 明天 Day 7 详细讲
```

### 3.2 强化学习 (Reinforcement Learning, RL) — 30min

```
核心思想: 通过试错和奖励学习

智能体 → 执行动作 → 环境给奖励 → 更新策略

关键要素:
  State s: 观察
  Action a: 动作
  Reward r: 奖励信号
  Policy π(a|s): 策略（就是我们的网络）
  Value V(s): 状态价值（预期未来奖励）

RL for Robotics 的挑战:
  - 样本效率低（需要大量试错）
  - 安全（机器人试错可能损坏）
  - 奖励设计难（什么算"做得好"？）
  - Sim-to-Real Gap（仿真里学的放现实可能不 work）

常用算法: PPO, SAC, DDPG, TD3
```

### 3.3 Sim-to-Real Transfer — 20min

```
问题: 仿真里训练的模型放到真实机器人上就不好使

原因:
  - 视觉差异 (仿真不够逼真)
  - 物理差异 (摩擦、质量、碰撞)
  - 噪声 (真实传感器有噪声)

解决方案:
  1. Domain Randomization:
     训练时随机化光照、纹理、物理参数
     模型被迫学到"本质"而不是"表面"

  2. 域适应 (Domain Adaptation):
     用少量真实数据微调

  3. 蒸馏 (Distillation):
     高保真仿真 → 教师模型 → 蒸馏 → 学生模型
```

---

## 模块 4: Open X-Embodiment 数据集（1 小时）

> ⭐ 这是 VLA 最重要的数据集，面试永远会问！

### 4.1 数据集概览 — 30min

```
Open X-Embodiment (OXE):
  至今最大的开源机器人数据集
  来自 34 个机器人实验室
  包含 60+ 个数据集
  超过 100 万条轨迹
  160 万条任务

目标: 做机器人的 "ImageNet 时刻"
  ImageNet → 推动了 CV 的革命
  OXE → 旨在推动机器人学习的革命

关键论文: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
```

### 4.2 数据格式 — 40min

```python
# OXE 的典型数据格式（RLDS 格式）
# 每条轨迹是一个 episode

trajectory = {
    "observation": {
        "image": ...,           # [T, H, W, C] 图像序列
        "image_primary": ...,   # 主摄像头
        "image_wrist": ...,     # 腕部摄像头
        "state": ...,           # [T, state_dim] 关节角度
    },
    "action": ...,              # [T, action_dim] 动作序列
    "language_instruction": "pick up the red block",  # 语言指令
    "language_embedding": ...,  # 预编码的语言特征（用通用句编码器）
}

# action 的格式（因数据集而异）:
# - 增量末端执行器控制: [dx, dy, dz, droll, dpitch, dyaw, gripper]
# - 绝对关节位置: [θ₁, ..., θ₆, gripper]
# - 通常归一化到 [-1, 1]
```

### 4.3 数据加载实践 — 20min

```python
import tensorflow_datasets as tfds

# 加载 OXE 数据（简化版）
# 实际使用需要 tfds 和特定的数据集包
# 这里展示概念

# 常用子集
datasets = [
    "fractal20220817_data",    # RT-1 的数据
    "bridge_dataset",          # Bridge v2 数据集
    "kuka",                    # Kuka 机械臂数据
    "berkeley_autolab_ur5",    # Berkeley UR5
]

# VLA 训练时如何混合多数据集:
# 1. 统一 action space（归一化）
# 2. 统一 observation space（resize 图像）
# 3. 统一 language embedding（用同一个编码器）
```

---

## 模块 5: 仿真环境 + MuJoCo 入门（1 小时）

### 5.1 主流仿真器 — 20min

| 仿真器 | 特点 | 适合场景 |
|--------|------|----------|
| **MuJoCo** | 物理引擎快、DeepMind 维护、gymnasium 接口 | RL 训练、VLA 仿真 |
| Isaac Sim | GPU 并行、光线追踪渲染 | 大规模视觉任务 |
| SAPIEN | 关节物体 | 开门、抽屉、工具 |
| Habitat | 室内导航 + 重排 | 移动操作 |
| Robosuite | 基于 MuJoCo、标准化 | 模仿学习、VLA 评估 |

### 5.2 快速上手 MuJoCo + gymnasium — 15min

```python
import gymnasium as gym

env = gym.make("HalfCheetah-v4")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, info = env.reset()

print(f"Obs dim: {env.observation_space.shape}")  # 位置+速度
print(f"Action dim: {env.action_space.shape}")     # 力矩
```

### 5.3 MuJoCo MJCF 模型格式入门 — 25min

> MJCF = MuJoCo XML Format, 是描述机器人/场景的标准格式

```xml
<!-- robot_arm.xml — 一个简单的 2 连杆机械臂模型 -->
<mujoco model="simple_arm">
  <compiler angle="radian"/>

  <worldbody>
    <!-- 地板 -->
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- 基座（固定） -->
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="0.1 0.05" rgba="0.5 0.5 0.5 1"/>

      <!-- 关节 1（旋转关节） -->
      <joint name="joint1" type="hinge" axis="0 0 1" range="-2.5 2.5"/>
      <!-- 连杆 1 -->
      <body name="link1" pos="0 0 0.05">
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0.5 0 0"
              size="0.03" rgba="0.3 0.7 0.3 1"/>

        <!-- 关节 2 -->
        <joint name="joint2" type="hinge" axis="0 0 1" range="-2.5 2.5"/>
        <!-- 连杆 2（含末端执行器） -->
        <body name="link2" pos="0.5 0 0">
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0.4 0 0"
                size="0.03" rgba="0.3 0.3 0.7 1"/>
          <!-- 末端标记（红色小球） -->
          <geom name="end_effector" type="sphere" pos="0.4 0 0"
                size="0.04" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 执行器（驱动关节的方式） -->
  <actuator>
    <motor name="motor1" joint="joint1" gear="100" ctrlrange="-2.5 2.5"/>
    <motor name="motor2" joint="joint2" gear="100" ctrlrange="-2.5 2.5"/>
  </actuator>
</mujoco>
```

```python
# 在 Python 中加载自定义 MJCF 模型
import mujoco
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("robot_arm.xml")
data = mujoco.MjData(model)

# 设置关节角度
data.qpos[0] = 0.5   # 关节 1: 0.5 rad
data.qpos[1] = -0.3  # 关节 2: -0.3 rad

# 计算正运动学（MuJoCo 自动算好！）
mujoco.mj_forward(model, data)

# 读取末端执行器的位置
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "end_effector")
ee_pos = data.geom_xpos[ee_id]
print(f"末端位置: {ee_pos}")

# 你可以用 MuJoCo 的计算结果验证你上面手写的 FK！
```

**MJCF 关键字段速查**:

| 字段 | 含义 | 例子 |
|------|------|------|
| `joint type="hinge"` | 旋转关节 | 机械臂关节 |
| `joint type="slide"` | 平移关节 | 滑轨 |
| `joint axis="0 0 1"` | 旋转轴 | 绕 Z 轴 |
| `joint range` | 关节限位 (rad) | `[-2.5, 2.5]` |
| `motor gear` | 力矩放大 | `gear="100"` |
| `geom type="capsule"` | 胶囊体（碰撞用） | 连杆 |
| `geom type="plane"` | 平面 | 地面/桌面 |
| `geom rgba` | 颜色（RGBA） | `"1 0 0 1"`=红色 |



---

## 🎤 费曼挑战（1 小时）

### 任务 1: 领域全景图 (20min)
画一张 A4 大小的思维导图，包含:
- 具身智能的三大模块（感知、决策、控制）
- 三种学习范式（IL, RL, Sim2Real）
- 主要 VLA 模型及其关系
- 关键数据集和仿真器

### 任务 2: 讲给新人 (20min)
假设你在给实验室新来的本科生介绍，用 5 分钟解释:
1. 具身智能和 ChatGPT 这种 AI 有什么本质不同？
2. 为什么机器人这么难？
3. VLA 能解决什么？

### 任务 3: 数据集分析 (20min)
去 [Open X-Embodiment 官网](https://robotics-transformer-x.github.io/)（或看论文的 Table 1），列出 5 个你最感兴趣的子数据集，说明:
- 什么机器人？
- 什么任务？
- action space 是什么样子？

---

## 📝 今日自检清单

- [ ] 我能解释 VLA 中 V、L、A 分别指什么
- [ ] 我理解 6-DOF 机械臂的含义
- [ ] 我知道关节空间和笛卡尔空间的区别
- [ ] 我能手写 2 连杆臂的正运动学公式
- [ ] 我能解释为什么 IK 比 FK 难
- [ ] 我知道 DH 参数是哪四个（a, α, d, θ）
- [ ] 我能解释雅可比矩阵的物理含义
- [ ] 我知道什么是奇异性及其对 VLA 部署的影响
- [ ] 我能读懂 MJCF XML 中的 joint 和 geom 定义
- [ ] 我能比较 IL、RL、Sim2Real 的优缺点
- [ ] 我知道 Open X-Embodiment 包含什么
- [ ] 我完成了全景思维导图

---

> [!info] 知识库关联
> - [[../../../01-基础理论/机器人运动学|机器人运动学]] — FK/IK/DH参数/雅可比/奇异性
> - [[../../../01-基础理论/3D视觉/3D视觉与点云|3D视觉与点云]] — 传感器 + 点云/PointNet
> - [[../../VLA方向综述|VLA方向综述]] — VLA领域全景
> - [[../../VLA模型总览|VLA模型总览]] — 现有VLA模型对比
>
> ✅ **完成打卡**: 具身智能全景已建立！明天深入行为克隆！
>
> 🔜 **明日预告**: Day 7 — 行为克隆与模仿学习，理解 BC 的数学本质和 ACT 的创新！
