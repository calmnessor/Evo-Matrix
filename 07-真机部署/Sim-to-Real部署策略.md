# Sim-to-Real 部署策略

> "仿真中 95% 的成功率为什么到真机上变成 30%？"——系统化弥合仿真与现实鸿沟的工程方法。

## Gap 诊断框架

```
Sim2Real Gap 两大来源:

1. 观测 Gap:
   仿真图像 ≠ 真实图像 (纹理、光照、噪点、运动模糊)
   → 策略看到的"世界"不一样

2. 动力学 Gap:
   仿真物理 ≠ 真实物理 (摩擦、阻尼、接触刚度、执行器延迟)
   → 同样的动作产生不同的结果

诊断方法: 用同一策略在仿真和真机上分别跑 N 次
  - 仿真高 + 真机低 = Sim2Real Gap (正常)
  - 仿真低 + 真机低 = 策略太差 (不是 Gap 问题)
  - 仿真低 + 真机高 = 仿真过于简化 (罕见)
```

## 1. 渐进部署策略

```
不要一步到位! 分四步走:

Step 1: 仿真达标
  在仿真中成功率 > 80% 再考虑上真机

Step 2: 基础标定验证
  确认视觉标定和运动学正确
  用硬编码已知位置验证

Step 3: 简化真机场景
  - 用仿真的相同物体 (YCB 标准物体)
  - 固定相机位置
  - 控制光照 (关窗、恒光源)
  → 最小化 Gap

Step 4: 逐步增加复杂度
  - 随机物体、新物体
  - 变化光照、背景
  - 动态环境
```

### 渐进 Check List

```python
def sim2real_checklist():
    checks = [
        # === 仿真阶段 ===
        ("仿真成功率 > 80%", lambda: sim_success_rate > 0.8),
        ("仿真中 5+ 种物体随机化", lambda: len(sim_objects) >= 5),
        ("仿真中有光照变化测试", lambda: sim_lighting_variation),

        # === 真机准备 ===
        ("手眼标定误差 < 1cm", lambda: hand_eye_error < 0.01),
        ("相机内参标定重投影 < 0.5px", lambda: reproj_error < 0.5),
        ("关节运动正常, 无异常响声", lambda: robot_motion_smooth),

        # === 简化真机 ===
        ("单个方块抓取成功率 > 80%", lambda: simple_grasp_rate > 0.8),
        ("策略输出频率 > 10Hz", lambda: control_freq > 10),

        # === 升级真机 ===
        ("3+ 种不同物体成功率 > 60%", lambda: multi_object_rate > 0.6),
        ("不同光照下成功率 > 50%", lambda: varied_lighting_rate > 0.5),
    ]
    return checks
```

## 2. Domain Randomization in Deployment

```
训练时 DR: 仿真参数随机化 → 策略鲁棒
部署时 DR 验证: 在真机上"模拟" DR

验证方法:
  □ 改变光照条件 (开/关灯、窗帘)
  □ 改变相机位置 (±5cm)
  □ 改变物体初始位置
  □ 添加背景杂物

如果策略在这些变化下仍稳定 → Sim2Real Gap 已缩到可接受范围
```

## 3. 在线适应

```python
"""
在线适应 = 真机上边跑边微调

方法 1: 在线校准
  看到物体 → 如果抓偏了 → 自动偏移修正 → 不重训策略

方法 2: 在线微调
  跑几个 episode → 收集数据 → 微调模型 → 继续跑
  (风险: 微调可能导致灾难性遗忘, 需要 KI 类保护)
"""

class OnlineAdapter:
    """在线抓取偏移校准"""

    def __init__(self):
        self.offset = np.zeros(3)  # 累积偏移

    def update(self, predicted_pose, actual_pose):
        """如果实际抓取位置与预测不一致 → 更新偏移"""
        error = actual_pose[:3] - predicted_pose[:3]
        # EMA 平滑
        self.offset = 0.9 * self.offset + 0.1 * error
        print(f"偏移更新: {self.offset * 1000:.1f} mm")

    def correct(self, predicted_pose):
        """应用偏移修正"""
        corrected = predicted_pose.copy()
        corrected[:3] += self.offset
        return corrected
```

## 4. 典型案例: π₀.₅ 的 Sim2Real 实践

```
π₀.₅ 的成功部署经验 (来自论文):

1. 数据多样性:
   - 100 个不同家庭收集的遥操作数据
   - 不是在一个仿真中训练的，而是在真实世界的多样性中

2. 视觉鲁棒性:
   - 训练时不做域随机化，但 VLM 预训练已提供视觉泛化
   - 关键: VLM 在数十亿张图片上见过各种光照、视角、背景

3. 物理鲁棒性:
   - Flow Matching 输出动作 → 不怕微小扰动 (比离散 token 鲁棒)
   - 用 RTC 异步推理 → 真实世界执行也平滑

4. 渐进部署:
   - 先在 100 个训练家庭中测试
   - 然后去 5 个全新家庭 → 测泛化
   - 失败 case 分析 → 针对改进 → 再加数据
```

## 5. 故障排查流程

```
真机跑崩了? 按这个顺序查:

1. 图像对吗?
   → 看相机画面, 确认清晰/无过曝/物体在视野内
   → 确认图像尺寸和训练时一致 (resize!)

2. 标定对吗?
   → 放一个已知位置的物体 → 相机看→机器人去碰 → 误差多少?
   → >2cm → 重新标定

3. 控制对吗?
   → 发一个简单的 1cm 步进 → 实测位移是否 = 1cm?
   → 检查动作空间: 末端增量还是关节位置? 和训练一致吗?

4. 频率对吗?
   → 推理+控制频率够不够? (>10Hz 至少, 50Hz 理想)
   → 延迟测试: 发动作到执行的时间差

5. 观测对吗?
   → 真机的观测分布和仿真一致吗?
   → 看 featur e 统计量 (mean/std) 是否漂移
```

## 6. 学习资源

| 资源 | 类型 | 内容 |
|------|------|------|
| **Domain Randomization 综述** | 论文 | "Domain Randomization for Sim-to-Real Transfer" |
| **Sim2Real 教程 (OpenAI)** | 博客 | Solving Rubik's Cube with a Robot Hand |
| **π₀.₅ 部署经验** | 论文 | arxiv.org/abs/2504.16054 — Section 5 |
| **CS 285 Sim2Real** | 课程 | Levine 讲解 Sim2Real 的核心方法 |

## 关联笔记
- [[../07-仿真与工具/仿真数据与Domain Randomization|Domain Randomization 详解]] — DR 原理与实现
- [[传感器标定与外参校准]] — Gap 诊断的关键前提
- [[常见问题排查]] — 真机失败的具体排查方法
- [[遥操作与数据收集]] — 真机数据回传改进模型
