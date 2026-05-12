# Docker 与可重复实验

> 让实验在任何机器上都能复现——Docker 封装环境，告别"在我机器上能跑"。

## 为什么具身智能研究需要 Docker？

```
场景 1: 论文复现
  "用作者提供的 requirements.txt → 依赖冲突 → 搞了一周没跑通"

场景 2: 多机器人部署
  "实验室有 3 台 Franka 工作站 → 每台环境不同 → 代码每台都要改"

场景 3: 合作开发
  "我在 Isaac Sim 2023.1 上写的代码 → 同事用的是 2024.2 → 不兼容"

场景 4: 论文提交
  "审稿人想跑我的代码 → 但没有 RTX 4090 → 跑不了"

Docker 解决所有这些:
  → 一次构建, 到处运行
  → 环境完全一致 (包括 CUDA 版本、驱动依赖)
  → 易于分享和复现
```

## Docker 基础

### 核心概念

```
Image (镜像):  环境的"蓝图" — 包含 OS、库、代码
Container (容器): 镜像的"实例" — 运行中的隔离环境
Dockerfile:     构建镜像的"配方"
Docker Compose: 多容器编排 (如 ROS 2 + VLA + 可视化)
Registry:       镜像仓库 (Docker Hub / NGC)
```

### Dockerfile 模板

```dockerfile
# VLA 训练环境 Dockerfile
FROM nvidia/cuda:12.1-cudnn8-devel-ubuntu22.04

# 系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    git wget curl \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 安装 PyTorch (CUDA 12.1)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 复制代码
WORKDIR /workspace
COPY . /workspace/

# 默认命令
CMD ["python", "train_vla.py"]
```

### 构建与运行

```bash
# 构建镜像
docker build -t vla-training:v1 .

# 运行 (带 GPU)
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    -it vla-training:v1 \
    python train_vla.py --batch_size 8

# 交互式运行
docker run --gpus all -it vla-training:v1 /bin/bash
```

## 具身智能专用 Docker 镜像

### NVIDIA NGC 镜像

```bash
# Isaac Sim
docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# PyTorch
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# ROS 2 + CUDA
docker pull nvcr.io/nvidia/isaac/ros:humble-ros_base-humble
```

### 常用 Docker Compose

```yaml
# docker-compose.yml — VLA 多容器开发环境
version: '3.8'

services:
  # VLA 推理服务
  vla:
    image: vla-inference:v1
    build:
      context: .
      dockerfile: Dockerfile.vla
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/models
      - ./checkpoints:/checkpoints
    network_mode: "host"
    command: python inference_server.py

  # ROS 2 Humble
  ros2:
    image: osrf/ros:humble-desktop
    network_mode: "host"
    volumes:
      - ./ros2_ws:/ros2_ws
    environment:
      - ROS_DOMAIN_ID=0
    command: ros2 launch vla_bringup vla.launch.py

  # 可视化 (Foxglove)
  foxglove:
    image: ghcr.io/foxglove/studio:latest
    network_mode: "host"
    ports:
      - "8080:8080"
```

## VLA 实验可重复性清单

### 1. 代码版本锁定

```bash
# 记录所有依赖的精确版本
pip freeze > requirements-lock.txt

# Git commit hash
git rev-parse HEAD > experiment_commit.txt

# Docker image digest
docker images --digests vla-training:v1
```

### 2. 随机种子管理

```python
import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 慢但确定
```

### 3. 数据版本管理

```bash
# 用 DVC (Data Version Control) 管理数据集
dvc init
dvc add data/bridge_v2/
dvc push  # 推到远程存储
git add data/bridge_v2.dvc
git commit -m "Bridge v2 dataset v1.0"
```

### 4. Experiment Tracking

```python
# Weights & Biases
import wandb
wandb.init(project="vla-training", config={
    "model": "openvla-7b",
    "dataset": "bridge_v2",
    "lora_rank": 16,
    "learning_rate": 2e-5,
    "docker_image": "vla-training:v1",
    "git_commit": "a1b2c3d",
})

# 或 MLflow
import mlflow
mlflow.set_experiment("vla-training")
mlflow.log_params({"model": "openvla-7b", ...})
mlflow.log_metric("success_rate", 0.72)

# 保存整个 Docker 镜像
mlflow.log_artifact("Dockerfile")
```

## 常见问题与解决

### GPU 不可用

```bash
# 确认 nvidia-docker 安装
nvidia-smi  # 宿主机确认 GPU
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi  # 容器内确认

# 如果失败:
# 安装 nvidia-container-toolkit
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### GUI 应用 (RViz/Isaac Sim)

```bash
# Linux: 允许 X11 转发
xhost +local:docker
docker run --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -it isaac-sim:latest
```

### 大镜像优化

```dockerfile
# 多阶段构建 → 减小最终镜像
FROM nvidia/cuda:12.1-devel AS builder
# 编译和构建...

FROM nvidia/cuda:12.1-runtime AS runtime
COPY --from=builder /build/output /app/
# 最终镜像不含编译工具 → 小很多

# 层合并 → 减少层数
RUN apt-get update && apt-get install -y pkg1 pkg2 pkg3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# 一个 RUN = 一层 (而不是三个 RUN = 三层)
```

## 自检问题

### 基础关
- [ ] 我理解 Docker Image 和 Container 的区别
- [ ] 我能写 Dockerfile 并构建镜像
- [ ] 我能用 docker run --gpus all 跑 GPU 任务

### 进阶关
- [ ] 我理解 Volume Mount (-v) 的用途
- [ ] 我能用 Docker Compose 编排多容器
- [ ] 我能优化 Dockerfile 减小镜像体积

### 实战关
- [ ] 我封装过 VLA 训练环境的完整 Docker 镜像
- [ ] 我在 Docker 中跑过 Isaac Sim 仿真
- [ ] 我把 Docker 镜像分享给了合作者且他们能直接跑通

## 关联笔记
- [[仿真与工具索引]] — 回到工具索引
- [[ROS2与机器人中间件]] — Docker + ROS 2 联合部署
- [[Isaac Sim与Isaac Lab]] — Isaac Sim Docker 镜像
- [[HuggingFace工具链]] — 模型和数据的版本管理
