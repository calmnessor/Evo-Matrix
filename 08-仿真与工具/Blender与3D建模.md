# Blender 与 3D 建模

> 机器人仿真的"美术工厂"——生成 3D 资产、合成训练数据、构建仿真场景。

## 在具身智能研究中的六大用途

| # | 用途 | 典型场景 |
|---|------|---------|
| 1 | **场景建模** | 搭建仿真环境（桌子、厨房、沙发等） |
| 2 | **合成数据** | 渲染不同视角、光照的 CV 训练数据 |
| 3 | **物体资产** | 创建 YCB 标准物体、自定义工具 |
| 4 | **Robot URDF 可视化** | 辅助可视化机械臂/机器人模型 |
| 5 | **Domain Randomization** | 批量生成变体纹理、光照、背景 |
| 6 | **3D Gaussian Splatting** | 真机场景 → 3DGS → 导入仿真 |

## 工具链概览

```
Blender (3D 建模/渲染)
    ↓ 导出
OBJ / STL / GLB / USD
    ↓ 导入
MuJoCo / Isaac Sim / PyBullet / Habitat
    ↓
仿真中的 VLA 训练/评估
```

### Blender 生态对比

| 工具 | 类型 | 用途 | 学习曲线 |
|------|------|------|---------|
| **Blender** | 开源 3D 软件 | 建模、渲染、动画 | 中等 |
| **Maya** | 商业 3D 软件 | 影视级建模 | 陡峭 |
| **3ds Max** | 商业 3D 软件 | 建筑可视化 | 中等 |
| **Houdini** | 商业程序化 | 程序化生成 (大量变体) | 陡峭 |
| **SketchUp** | 轻量建模 | 快速场景搭建 | 平缓 |
| **Meshlab** | 开源处理 | 点云/网格处理 | 平缓 |

## 工作流 1: 创建仿真场景

### Blender → MuJoCo

```
步骤:
1. Blender 中建模场景 (桌子、物体)
2. 导出为 OBJ/STL
3. 编写 MJCF XML，引用导出的 mesh
4. 在 MuJoCo 中加载
```

```xml
<!-- MJCF 引用 Blender 导出的 mesh -->
<asset>
  <mesh name="table" file="table.obj" scale="1 1 1"/>
  <mesh name="cup" file="cup.stl"/>
</asset>

<worldbody>
  <body name="table_body" pos="0 0 0">
    <geom name="table_geom" type="mesh" mesh="table" rgba="0.8 0.7 0.5 1"/>
  </body>
  <body name="cup_body" pos="0 0 0.8">
    <geom name="cup_geom" type="mesh" mesh="cup" rgba="1 0.2 0.2 1"/>
    <freejoint/>  <!-- 自由物体 -->
  </body>
</worldbody>
```

### Blender → Isaac Sim

Isaac Sim 原生支持 USD (Universal Scene Description) 格式：

```
Blender → Export USD → Isaac Sim (直接加载,支持材质/动画)
```

USD 比 OBJ/STL 保留更多信息（材质、光照、层级结构），是 Isaac Sim 的推荐格式。

### Blender → PyBullet

```python
import pybullet as p

# 直接加载 OBJ (需要单独定义碰撞体)
p.loadURDF("robot.urdf")  # URDF 引用 OBJ/STL mesh

# 也可以通过 SDFormat
```

## 工作流 2: 合成数据生成

### 批量渲染 (bpy 脚本)

```python
import bpy
import numpy as np

# 批量渲染不同视角和光照
def batch_render_scene(
    object_name,
    camera_positions,  # [(x,y,z, rx,ry,rz), ...]
    lighting_conditions,  # [energy_value, ...]
    output_dir="./renders/"
):
    obj = bpy.data.objects[object_name]
    cam = bpy.data.objects["Camera"]
    light = bpy.data.objects["Light"]

    for i, (pos, rot) in enumerate(camera_positions):
        # 设置相机位置
        cam.location = pos[:3]
        cam.rotation_euler = rot[3:] if len(rot) > 3 else (0, 0, 0)

        for j, light_energy in enumerate(lighting_conditions):
            light.data.energy = light_energy

            # 设置输出
            bpy.context.scene.render.filepath = f"{output_dir}/img_{i}_{j}.png"
            bpy.ops.render.render(write_still=True)

# 使用示例
camera_positions = [
    (1.5, 0, 1.0, 0, 0, 1.57),   # 正面
    (0, 1.5, 1.0, 0, 0, 3.14),   # 侧面
    (1.0, 1.0, 1.5, 0, 0, 2.0),  # 斜上方
]
lighting = [100, 200, 500, 1000]  # 不同亮度
batch_render_scene("Cup", camera_positions, lighting)
```

### Domain Randomization 进阶

```python
import bpy
import random

def domain_randomization(num_scenes=1000):
    """生成随机场景变体用于 sim-to-real"""
    for i in range(num_scenes):
        # 随机替换背景
        bpy.data.worlds["World"].node_tree.nodes["Background"] \
           .inputs[1].default_value = random.choice(bg_colors)

        # 随机纹理
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and random.random() > 0.5:
                mat = obj.active_material
                if mat:
                    # 随机颜色偏移
                    mat.node_tree.nodes["Principled BSDF"] \
                       .inputs[0].default_value = (
                           random.random(),
                           random.random(),
                           random.random(),
                           1.0
                       )

        # 随机物体位置
        for obj_name in ["cup", "plate", "spoon"]:
            if obj_name in bpy.data.objects:
                obj = bpy.data.objects[obj_name]
                obj.location = (
                    random.uniform(-0.3, 0.3),
                    random.uniform(-0.3, 0.3),
                    obj.location.z,
                )

        # 渲染
        bpy.context.scene.render.filepath = f"./dr_output/{i:04d}.png"
        bpy.ops.render.render(write_still=True)
```

### 分割 Mask 渲染

```python
# 渲染语义分割 mask（用于 VLA 的视觉理解）
def render_segmentation_masks():
    # 为每个物体设置不同的 pass_index
    for i, obj in enumerate(bpy.data.objects):
        obj.pass_index = i + 1

    # 启用 Object Index pass
    bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True

    # 渲染
    bpy.ops.render.render(write_still=True)

    # 在 Compositor 中用 ID Mask 节点提取每个物体的 mask
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    # ... 添加 ID Mask 节点 → File Output 节点
    # 最终输出: mask_001.png (杯子), mask_002.png (盘子), ...
```

## 工作流 3: 机器人 URDF 可视化

```python
# Blender 中导入 URDF 进行可视化
# 需要安装: https://github.com/omri1348/urdf-blender-import

import bpy
from urdf_blender_import import URDFImporter

importer = URDFImporter()
importer.load("./franka_panda.urdf")

# 现在可以:
# 1. 调整材质让机器人更好看
# 2. 添加环境物体
# 3. 渲染机器人在不同姿势下的图像
# 4. 生成带机器人的训练数据
```

## 工作流 4: 3DGS → 仿真

```
真实场景
    ↓ 多视角照片
3D Gaussian Splatting (3DGS) 重建
    ↓ 导出 3DGS
→ 可在 Blender 中渲染新视角
→ 可导出为 mesh → 导入仿真器
→ 得到"照片级真实"的仿真场景
```

这是当前 Sim-to-Real 的前沿方向——仿真器中的场景来自真实 3D 重建，不再是手工建模。

## 常用 Blender 插件

| 插件 | 用途 | 获取 |
|------|------|------|
| **URDF Importer** | 导入 URDF 可视化机器人 | GitHub |
| **CAD Transform** | 精确 CAD 风格变换 | 内置 |
| **Bool Tool** | 布尔运算 (快速建模) | 内置 |
| **Tissue** | 程序化几何 | 内置 |
| **Mega Scans** | 照片级材质库 (免费) | Quixel |
| **Poly Haven** | 免费 HDRI/材质/模型 | polyhaven.com |

## 课题组现有资源

| 资源 | 位置 | 状态 |
|------|------|------|
| Blender 主程序 | `jushenzhineng/Blender/` | ✅ |
| Blender 插件 | `jushenzhineng/Blender插件/` | ✅ |
| 3D 模型库 | 待整理 | — |
| 合成数据脚本 | 待开发 | — |

## 学习路径

### 入门 (第 1 周)
1. 安装 Blender → 熟悉基本操作 (移动/旋转/缩放/渲染)
2. 创建一个简单的桌面场景 (桌子 + 几个物体)
3. 导出 OBJ → 导入 MuJoCo → 加载成功

### 进阶 (第 2-3 周)
4. 学习 bpy Python API → 写第一个批量渲染脚本
5. 导入一个 URDF 机器人模型 → 可视化
6. 制作 Domain Randomization 脚本 → 生成 100 个场景变体

### 实战 (第 4+ 周)
7. 创建完整 VLA 训练用的仿真场景
8. 批量生成带标注的合成数据 (RGB + 分割 mask + 深度)
9. 真机场景 → 3DGS → Blender → 仿真器 全流程

## 自检问题

### 基础关
- [ ] 我能用 Blender 创建简单的桌面场景（桌子 + 几个几何体）
- [ ] 我知道如何导出 OBJ/STL/USD 文件
- [ ] 我能将 Blender 导出的 mesh 导入 MuJoCo/Isaac Sim

### 进阶关
- [ ] 我理解 bpy Python API，能写自动化脚本
- [ ] 我能批量渲染不同视角和光照的图像
- [ ] 我理解 Domain Randomization 的原理和实现

### 实战关
- [ ] 我生成过完整的合成训练数据（100+ 场景变体）
- [ ] 我做过 3DGS 重建 → Blender → 仿真的 pipeline
- [ ] 我创建过完整的 VLA 仿真训练场景

## 关联笔记
- [[MuJoCo]] — Blender 导出的 mesh 在 MuJoCo 中的使用
- [[Isaac Sim与Isaac Lab]] — USD workflow 与光追渲染
- [[仿真数据与Domain Randomization]] — 合成数据的深入讨论
- [[../../01-基础理论/3D视觉与点云]] — 3DGS 和点云处理
- [[仿真与工具索引]] — 回到工具索引
