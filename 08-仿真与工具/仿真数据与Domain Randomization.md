# 仿真数据与 Domain Randomization

> 让仿真中学到的策略能迁移到真实世界——Sim-to-Real 转移的核心技术。

## 核心问题：Sim-to-Real Gap

```
为什么仿真中 95% 成功率的策略放到真机上只有 30%？

三大 Gap:
  1. 视觉 Gap:    仿真渲染 ≠ 真实相机图像 (纹理、光照、噪点)
  2. 物理 Gap:    仿真物理 ≠ 真实物理 (摩擦、阻尼、接触刚度)
  3. 行为 Gap:    仿真中的策略学会了"利用仿真 bug"
                 (如穿透物体 → 真实世界做不到)
```

## Domain Randomization (DR) — 核心解决方案

### 基本原理

```
不追求仿真 = 真实 (太难了)
而是: 仿真中见过各种变化 → 真实世界只是其中一种变化

具体: 每次 reset 随机化仿真参数
  → 策略学会对参数变化鲁棒
  → 真实世界在随机化范围内 → 策略也能处理
```

### 可随机化的维度

| 类别 | 参数 | 范围示例 |
|------|------|---------|
| **视觉** | 光照强度/颜色 | 50-200% 变化 |
| | 物体纹理/颜色 | 全部 HSV 空间 |
| | 相机噪点 | 高斯噪声 σ=[0, 0.05] |
| | 背景替换 | 随机真实照片背景 |
| | 相机位姿 | ±5cm, ±5° |
| **物理** | 物体质量 | 50-200% 标称值 |
| | 摩擦系数 | [0.3, 1.5] |
| | 关节阻尼 | [0.5, 2.0] × 标称值 |
| | 执行器力矩 | [0.8, 1.2] × 标称值 |
| | 时间步长 | [0.9, 1.1] × 仿真 dt |
| **几何** | 物体尺寸 | ±20% 缩放 |
| | 物体初始位置 | 在桌子范围内均匀随机 |
| | 机器人基座位置 | ±10cm |
| **动态** | 观测延迟 | [0, 50]ms |
| | 动作噪声 | 高斯噪声 σ=[0, 0.02] |
| | 外力扰动 | 随机碰触/推动 |

### MuJoCo 中实现 DR

```python
import numpy as np
import mujoco

def randomize_physics(model):
    """随机化物理参数"""
    # 质量
    model.body_mass[:] *= np.random.uniform(0.8, 1.2, model.nbody)

    # 摩擦
    model.geom_friction[:] *= np.random.uniform(0.7, 1.3, (model.ngeom, 3))

    # 阻尼
    model.dof_damping[:] *= np.random.uniform(0.5, 2.0, model.nv)

    # 关节限位 (稍微放松)
    model.jnt_range[:] *= np.random.uniform(0.95, 1.05, (model.njnt, 2))

    # 执行器 gear
    model.actuator_gear[:, 0] *= np.random.uniform(0.9, 1.1, model.nu)

def randomize_visuals(model, data):
    """随机化视觉参数"""
    # 光照
    model.vis.headlight.ambient[:] = np.random.uniform(0.2, 0.6, 3)
    model.vis.headlight.diffuse[:] = np.random.uniform(0.4, 0.8, 3)

    # 物体颜色
    for i in range(model.ngeom):
        rgba = model.geom_rgba[i]
        # 随机色相偏移
        model.geom_rgba[i][:3] = rgba[:3] * np.random.uniform(0.7, 1.3, 3)
        model.geom_rgba[i] = np.clip(model.geom_rgba[i], 0, 1)

def randomize_init_state(model, data, object_body_ids):
    """随机化初始状态"""
    mujoco.mj_resetData(model, data)

    # 随机化物体位置
    for obj_id in object_body_ids:
        # 在桌面上方随机放置
        x = np.random.uniform(-0.3, 0.3)
        y = np.random.uniform(-0.3, 0.3)
        z = np.random.uniform(0.02, 0.15)  # 桌面上方

        # 设置自由物体的位置
        qpos_addr = model.body_jntadr[obj_id]
        data.qpos[qpos_addr:qpos_addr+3] = [x, y, z]

        # 随机朝向 (只改变 yaw)
        yaw = np.random.uniform(0, 2 * np.pi)
        data.qpos[qpos_addr+3:qpos_addr+7] = [0, 0, np.sin(yaw/2), np.cos(yaw/2)]

    mujoco.mj_forward(model, data)
```

### Isaac Sim 中的 DR

Isaac Sim 有**可视化 DR 工具**，可以拖拽式配置随机化范围和分布：

```python
from omni.isaac.dr import DomainRandomizer

# 创建 DR 管理器
dr = DomainRandomizer()

# 光照随机化
dr.add_light_randomizer(
    light_path="/World/Lights/DomeLight",
    intensity_range=(500, 2000),
    color_temperature_range=(3000, 8000),
)

# 纹理随机化
dr.add_material_randomizer(
    prim_paths=["/World/Table", "/World/Objects/*"],
    color_range=[(0, 0, 0), (1, 1, 1)],
    roughness_range=(0.1, 0.9),
    metallic_range=(0.0, 0.5),
)

# 相机随机化
dr.add_camera_randomizer(
    camera_path="/World/Camera",
    position_noise=0.02,  # ±2cm
    rotation_noise=1.0,   # ±1°
)

# 应用随机化
dr.apply()
```

## DR 进阶策略

### 1. 课程式 Domain Randomization (Curriculum DR)

```python
# 从简单开始，渐进增加难度
def curriculum_dr(difficulty, max_diff=1.0):
    """难度从 0 (无随机化) 到 1.0 (最大随机化)"""
    range_multiplier = difficulty / max_diff  # [0, 1]

    mass_range = 1.0 + np.array([-0.1, 0.1]) * range_multiplier
    # difficulty=0.0: mass = 1.0 ± 0%  (无变化)
    # difficulty=0.5: mass = 1.0 ± 5%
    # difficulty=1.0: mass = 1.0 ± 10%

    return mass_range

# 训练循环
for epoch in range(total_epochs):
    difficulty = epoch / total_epochs  # 线性增长
    dr_params = curriculum_dr(difficulty)
    train_with_dr(dr_params)
```

### 2. 自适应 Domain Randomization (ADR)

```python
# 根据策略性能自动调整 DR 范围
# 太容易 → 扩大范围, 太难 → 缩小范围

class AdaptiveDR:
    def __init__(self, initial_ranges, target_success=0.5):
        self.ranges = initial_ranges  # {param: (low, high)}
        self.target_success = target_success

    def update(self, recent_success_rate):
        """如果成功率 > 目标 → 扩大 DR (更难)"""
        if recent_success_rate > self.target_success:
            for param in self.ranges:
                low, high = self.ranges[param]
                self.ranges[param] = (low * 0.9, high * 1.1)
        else:
            # 成功率 < 目标 → 缩小 DR (更简单)
            for param in self.ranges:
                low, high = self.ranges[param]
                self.ranges[param] = (low * 1.1, high * 0.9)
```

### 3. 结构化 Domain Randomization

```python
# 不是独立随机每个参数，而是按物理规律耦合
def structured_dr():
    # 大物体 → 重 (ρ 保持合理)
    scale = np.random.uniform(0.8, 1.2)
    mass = scale ** 3 * nominal_mass  # ∝ 体积

    # 粗糙表面 → 高摩擦 + 高阻尼
    roughness = np.random.uniform(0, 1)
    friction = 0.3 + roughness * 0.9  # [0.3, 1.2]
    damping = 0.1 + roughness * 0.4   # [0.1, 0.5]

    # 光照和相机设在一起
    is_dark = np.random.binomial(1, 0.3)
    if is_dark:
        light_intensity = np.random.uniform(50, 200)
        camera_gain = np.random.uniform(2, 4)
    else:
        light_intensity = np.random.uniform(500, 2000)
        camera_gain = 1.0
```

## 数据增强 (视觉 DR 的补充)

```python
import albumentations as A

# 在仿真渲染图像上做增强 ≠ 替代 DR，而是补充
# DR = 物理参数变化 (改变场景本身)
# 数据增强 = 图像后处理 (给 VLA 额外的鲁棒性)

robot_aug = A.Compose([
    # 颜色抖动
    A.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.2, hue=0.05, p=0.5),
    # 高斯噪点 (模拟相机噪点)
    A.GaussNoise(var_limit=(0, 20.0), p=0.3),
    # 模糊 (模拟运动模糊 / 失焦)
    A.MotionBlur(blur_limit=3, p=0.1),
    A.GaussianBlur(blur_limit=3, p=0.1),
    # 随机裁剪/缩放 (模拟相机位姿变化)
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0), p=0.3),
    # 随机遮挡 (模拟机器人本体遮挡)
    A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
])

# 重要: 不要水平翻转! 机器人场景有物理左右
# 重要: 不要太大旋转! 重力方向不应变
```

## Sim-to-Real 评估协议

```python
def evaluate_sim2real_gap(sim_env, real_robot, policy, num_trials=50):
    """定量评估 Sim-to-Real Gap"""

    # Step 1: 仿真评估
    sim_success = 0
    for _ in range(num_trials):
        sim_obs = sim_env.reset()
        sim_success += run_episode(sim_env, policy, sim_obs)

    # Step 2: 真机评估
    real_success = 0
    for _ in range(num_trials):
        real_obs = real_robot.reset()
        real_success += run_episode(real_robot, policy, real_obs)

    sim_rate = sim_success / num_trials
    real_rate = real_success / num_trials

    gap = sim_rate - real_rate
    print(f"Sim Success: {sim_rate:.2%}")
    print(f"Real Success: {real_rate:.2%}")
    print(f"Sim2Real Gap: {gap:+.2%}")

    if gap > 0.3:
        print("⚠️ Large Sim2Real gap — increase DR!")
    return gap
```

## 自检问题

### 基础关
- [ ] 我理解 Sim-to-Real Gap 的三大来源（视觉/物理/行为）
- [ ] 我知道 Domain Randomization 的基本原理
- [ ] 我能列举至少 5 种可随机化的参数

### 进阶关
- [ ] 我理解 Curriculum DR 和 ADR 的区别
- [ ] 我知道为什么结构化 DR 比独立随机更有效
- [ ] 我能区分 DR（改变场景）和数据增强（改变图像）

### 实战关
- [ ] 我实现过完整的 DR pipeline (物理 + 视觉 + 几何)
- [ ] 我做过 Sim-to-Real 转移并定量评估 Gap
- [ ] 我调优过 DR 范围使 Sim2Real Gap <15%

## 关联笔记
- [[仿真与工具索引]] — 回到工具索引
- [[MuJoCo]] — MuJoCo 中的 DR 实现
- [[Isaac Sim与Isaac Lab]] — Isaac Sim 的 DR 工具
- [[../../01-基础理论/强化学习]] — Sim2Real 在 RL 中的应用
