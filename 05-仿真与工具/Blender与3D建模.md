# Blender 与 3D 建模

> 机器人仿真的"美术工厂"——生成 3D 资产和合成数据

## 在机器人研究中的用途

1. **场景建模**: 搭建仿真环境（桌子、物体、房间）
2. **合成数据**: 渲染不同视角、光照的 CV 训练数据
3. **Robot URDF**: 辅助可视化机械臂/机器人模型
4. **Domain Randomization**: 批量生成变体纹理和光照

## 课题组现有资源

| 资源 | 位置 |
|------|------|
| Blender 主程序 | `jushenzhineng/Blender/` |
| Blender 插件 | `jushenzhineng/Blender插件/` |

## Blender → MuJoCo 工作流

```
Blender 建模 → 导出为 MJCF/STL/OBJ → MuJoCo 加载
```

MJCF 可以直接引用 Blender 导出的 mesh 文件。

## Python 脚本 (bpy)

```python
import bpy

# Blender 可以用 Python 脚本批量操作
# 批量渲染不同视角:
for angle in range(0, 360, 30):
    bpy.data.objects["Camera"].rotation_euler[2] = angle * 3.14159 / 180
    bpy.ops.render.render(write_still=True)
```

## 自检问题
- [ ] 我能用 Blender 创建一个简单的桌面场景
- [ ] 我知道如何导出模型供 MuJoCo 使用

## 关联笔记
- [[MuJoCo]] — Blender 资产的仿真目的地
- [[3D视觉与点云]] — 3D 研究
- [[05-仿真与工具/仿真与工具索引|仿真工具索引]]
