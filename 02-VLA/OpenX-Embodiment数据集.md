# Open X-Embodiment 数据集

> 机器人学的"ImageNet 时刻"——最大的开源机器人数据集

## 数据规模

- 来自 34 个机器人实验室
- 60+ 个子数据集
- 100+ 万条轨迹
- 160+ 万条任务

## 核心论文

"Open X-Embodiment: Robotic Learning Datasets and RT-X Models"

**思想**: ImageNet 推动了 CV 革命，OXE 旨在推动机器人学习革命。

## 数据格式 (RLDS)

```python
trajectory = {
    "observation": {
        "image": ...,           # [T, H, W, C] 图像序列
        "image_primary": ...,   # 主摄像头
        "image_wrist": ...,     # 腕部摄像头
        "state": ...,           # [T, state_dim] 关节角度
    },
    "action": ...,              # [T, action_dim] 动作序列
    "language_instruction": "pick up the red block",
    "language_embedding": ...,  # 预编码语言特征
}
```

## Action 格式（因数据集而异）

| 类型 | 格式 | 示例 |
|------|------|------|
| 增量末端控制 | [dx, dy, dz, dr, dp, dy, gripper] | RT-1, RT-2 |
| 绝对关节位置 | [θ₁, ..., θ₆, gripper] | ACT |
| 归一化范围 | 通常 [-1, 1] | |

## 重要子集

| 数据集 | 机器人 | 任务类型 | 特点 |
|--------|--------|---------|------|
| Fractal | Everyday Robots | 桌面操作 | RT-1 的数据 |
| Bridge v2 | WidowX | 桌面操作 | 多视角，质量高 |
| Kuka | Kuka IIWA | 工业操作 | 精确控制 |
| Berkeley UR5 | UR5 | 抓取 | 基础操作 |
| Stanford | Franka Panda | 精细操作 | 科研质量 |

## 训练 VLA 时的数据处理

```python
# 多数据集混合训练的关键步骤
1. 统一 action space (归一化到 [-1, 1])
2. 统一 observation space (resize 图像到统一尺寸)
3. 统一 language embedding (用同一个编码器)
4. 数据采样策略: 平衡 vs 按规模加权
```

## 自检问题
- [ ] 我知道 OXE 包含多少数据量和来源
- [ ] 我理解 OXE 的数据格式（RLDS）
- [ ] 我知道混合多数据集时需要做哪些标准化
- [ ] 我能列举至少 3 个 OXE 子集及其特点

## 关联笔记
- [[VLA模型总览]] — OXE 是所有 VLA 的训练数据
- [[数据集索引]] — 更多公开数据集
- [[行为克隆]] — 训练方式
- [[05-仿真与工具/仿真与工具索引|仿真工具]] — 数据不足时的补充
