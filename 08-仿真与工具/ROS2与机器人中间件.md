# ROS 2 与机器人中间件

> 机器人软件开发的操作系统——ROS 2 是连接 VLA 模型和真实机器人的桥梁。

## 什么是 ROS？

ROS (Robot Operating System) 不是真正的操作系统，而是**机器人软件开发的中间件和工具集**：

```
ROS = 通信中间件 + 标准消息格式 + 工具生态 + 包管理

核心价值:
  1. 模块化: 每个功能是一个独立节点 (Node)
  2. 通信: 节点间通过话题 (Topic) / 服务 (Service) / 动作 (Action) 通信
  3. 标准化: 传感器、执行器、算法的统一接口
  4. 生态: 大量现成的驱动、算法、可视化工具
```

## ROS 1 vs ROS 2

| | ROS 1 | ROS 2 |
|---|-------|-------|
| **通信** | 自定义 TCP/UDP | DDS (工业标准) |
| **OS** | Linux only | Linux / Windows / macOS |
| **实时性** | 困难 | ✅ 原生支持 |
| **多机器人** | 困难 | ✅ 原生支持 |
| **安全** | 无 | ✅ DDS Security |
| **Python** | 2.7 / 3.6 | 3.6+ |
| **维护** | 2025 EOL | 活跃开发中 |
| **VLA 推荐** | ❌ | ✅ |

**关键**: 新项目用 ROS 2。ROS 1 将在 2025 年停止维护。

## ROS 2 核心概念

### Node (节点)

```python
import rclpy
from rclpy.node import Node

class VLAInferenceNode(Node):
    def __init__(self):
        super().__init__('vla_inference')
        # 订阅图像话题
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw',
            self.image_callback, 10
        )
        # 发布动作话题
        self.action_pub = self.create_publisher(
            JointTrajectory, '/arm/joint_trajectory', 10
        )
        # 加载 VLA 模型
        self.vla_model = load_model("openvla-7b")

    def image_callback(self, msg):
        # VLA 推理 → 发布动作
        action = self.vla_model.predict(msg)
        self.action_pub.publish(action)
```

### Topic / Service / Action

```
Topic (话题) — 单向流式数据:
  发布者 → [ /camera/image ] → 订阅者
  使用: 图像、传感器数据、关节状态
  频率: 10-100Hz

Service (服务) — 请求/响应:
  客户端 → [Request] → 服务端 → [Response]
  使用: IK 解算、运动规划请求
  频率: 低频，按需调用

Action (动作) — 长时间目标 + 反馈:
  客户端 → [Goal] → 服务端 → [Feedback...] → [Result]
  使用: "移动到目标位置" — 执行过程有进度反馈
  频率: 一次调用，长时间执行
```

### Launch 文件

```python
# Python launch 文件 (ROS 2 推荐)
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 相机驱动
        Node(package='realsense2_camera',
             executable='realsense2_camera_node',
             name='camera'),
        # VLA 推理
        Node(package='vla_ros2',
             executable='vla_inference',
             name='vla',
             parameters=[{'model_path': '/models/openvla-7b'}]),
        # 机器人控制
        Node(package='robot_control',
             executable='arm_controller',
             name='arm'),
    ])
```

## VLA 部署架构

```
┌────────────────────────────────────────────────┐
│                   VLA 服务器                     │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │ 视觉编码  │   │ VLM 推理 │   │ 动作生成    │  │
│  │ SigLIP   │→  │ Gemma    │→  │ Flow Match │  │
│  └──────────┘   └──────────┘   └────────────┘  │
│        ↑                              ↓         │
└────────┼──────────────────────────────┼─────────┘
         │ ROS 2 Topic                  │ ROS 2 Topic
         │ /camera/image                │ /arm/command
         ↓                              ↓
┌─────────────────┐          ┌─────────────────────┐
│   相机驱动节点    │          │   机器人控制节点      │
│  RealSense/Zed  │          │  Franka/UR/Kinova   │
└─────────────────┘          └─────────────────────┘
```

### 完整 VLA 推理节点

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import torch
from cv_bridge import CvBridge

class VLAInferenceNode(Node):
    def __init__(self):
        super().__init__('vla_inference_node')

        # ROS 参数
        self.declare_parameter('model_path', '/models/openvla-7b')
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('action_horizon', 50)

        # 加载 VLA 模型
        model_path = self.get_parameter('model_path').value
        self.get_logger().info(f'Loading VLA model from {model_path}...')
        self.vla_model = self.load_vla(model_path)

        # CV Bridge (ROS Image → numpy)
        self.bridge = CvBridge()

        # 订阅
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw',
            self.image_callback, 1
        )
        self.joint_sub = self.create_subscription(
            JointState, '/arm/joint_states',
            self.joint_callback, 1
        )

        # 发布
        self.action_pub = self.create_publisher(
            JointTrajectory, '/arm/joint_trajectory_command', 10
        )

        # 状态缓存
        self.latest_image = None
        self.latest_joints = None

        # 推理定时器
        dt = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(dt, self.inference_loop)

        # 动作块缓存
        self.action_buffer = None
        self.buffer_idx = 0

        self.get_logger().info('VLA Inference Node ready!')

    def load_vla(self, path):
        """加载 VLA 模型"""
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        return model

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def joint_callback(self, msg):
        self.latest_joints = np.array(msg.position)

    def inference_loop(self):
        """主推理循环"""
        if self.latest_image is None or self.latest_joints is None:
            return

        # 需要新推理?
        if self.action_buffer is None or self.buffer_idx >= len(self.action_buffer):
            # VLA 推理
            with torch.no_grad():
                actions = self.vla_model.predict(
                    image=self.latest_image,
                    proprio=self.latest_joints,
                    instruction="pick up the red cup",  # 可从外部 topic 获取
                )
            self.action_buffer = actions  # [H, act_dim]
            self.buffer_idx = 0
            self.get_logger().debug(f'New action chunk: {len(actions)} steps')

        # 发布当前动作
        action = self.action_buffer[self.buffer_idx]
        self.buffer_idx += 1
        self.publish_action(action)

    def publish_action(self, action):
        """发布关节轨迹"""
        msg = JointTrajectory()
        msg.joint_names = [f'panda_joint{i+1}' for i in range(7)]
        point = JointTrajectoryPoint()
        point.positions = action[:7].tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 20000000  # 20ms = 50Hz
        msg.points = [point]
        self.action_pub.publish(msg)
```

## 必备工具

### RViz2 — 可视化

```bash
rviz2  # 显示机器人状态、传感器数据、TF 树
```

### ros2_control — 机器人控制框架

```bash
# ros2_control 是 ROS 2 的标准控制框架
# 提供标准化的控制器接口 (JointTrajectoryController 等)
sudo apt install ros-humble-ros2-control
```

### MoveIt 2 — 运动规划

```bash
# MoveIt 2 = 运动规划 + 碰撞避免 + 抓取规划
sudo apt install ros-humble-moveit
```

```python
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy

# 初始化 MoveIt
robot = MoveItPy(node_name="vla_moveit")
arm = robot.get_planning_component("panda_arm")

# VLA 输出目标位姿 → MoveIt 规划无碰撞路径
arm.set_start_state_to_current_state()
arm.set_goal_state(pose_stamped_msg=target_pose)
plan_result = arm.plan()
if plan_result:
    robot.execute(plan_result.trajectory, controllers=[])
```

### ros2 bag — 数据记录与回放

```bash
# 记录所有话题
ros2 bag record -a -o my_experiment

# 选择性记录
ros2 bag record /camera/image /arm/joint_states -o vla_data

# 回放
ros2 bag play vla_data
```

### Foxglove — Web 可视化

Foxglove Studio 是 ROS 2 的 Web 端可视化工具，支持实时监控、数据回放、自定义面板。

## 关键通信延迟

```
VLA 延迟预算 (目标: <100ms):
  ┌─────────────────────────────────────┐
  │ 相机采集:       ~10ms               │
  │ ROS 传输:        ~5ms               │
  │ VLA 推理:       ~50-200ms (瓶颈!)   │
  │ ROS 传输:        ~5ms               │
  │ 控制器执行:      ~5ms               │
  │ ─────────────────────               │
  │ 总计:           ~75-225ms           │
  └─────────────────────────────────────┘

优化:
  1. TensorRT/ONNX 加速推理
  2. 降低图像分辨率 (640→320)
  3. 用 RTC 消除推理等待
  4. Zero-copy 传输 (shared memory)
```

## 自检问题

### 基础关
- [ ] 我理解 ROS 2 的核心概念（Node/Topic/Service/Action）
- [ ] 我知道 ROS 1 和 ROS 2 的主要区别
- [ ] 我能写一个简单的订阅/发布节点

### 进阶关
- [ ] 我能用 Launch 文件启动多节点系统
- [ ] 我理解 ros2_control 和 MoveIt 的定位
- [ ] 我能用 ros2 bag 记录和回放实验数据

### 实战关
- [ ] 我搭建过完整的 VLA→ROS 2→真机 pipeline
- [ ] 我解决过 ROS 2 通信延迟问题
- [ ] 我集成过 MoveIt 做 VLA 的安全运动规划

## 关联笔记
- [[仿真与工具索引]] — 回到工具索引
- [[Docker与可重复实验]] — ROS 2 环境封装
- [[Isaac Sim与Isaac Lab]] — Isaac Sim 的 ROS 2 Bridge
- [[../../02-VLA/VLA模型总览]] — VLA 的真机部署
