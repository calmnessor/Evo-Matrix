# Python 环境管理

> 避免"在我机器上能跑"问题——用版本管理和环境隔离

## Conda

```bash
# 创建环境
conda create -n vla python=3.10 -y
conda activate vla

# 导出环境（精确版本）
conda env export > environment.yml

# 从文件恢复
conda env create -f environment.yml

# 列出环境
conda env list
```

## pip + virtualenv

```bash
# 创建虚拟环境
python -m venv vla_env
source vla_env/bin/activate  # Linux/Mac
vla_env\Scripts\activate     # Windows

# 依赖管理
pip install -r requirements.txt
pip freeze > requirements.txt
```

## CUDA 与 PyTorch

```bash
# 检查 CUDA 版本
nvidia-smi
nvcc --version

# 安装匹配的 PyTorch
# 去 https://pytorch.org/get-started/locally/ 查对应命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 课题组环境

```bash
conda create -n vla python=3.10 -y
conda activate vla
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft
pip install numpy matplotlib jupyter tqdm wandb
pip install gymnasium mujoco robosuite
pip install tensorflow tensorflow-datasets  # OXE 数据加载
```

## 常见问题

| 问题 | 解决 |
|------|------|
| CUDA out of memory | 减小 batch size 或用 gradient accumulation |
| 版本冲突 | 用 Conda 创建全新环境 |
| pip 安装慢 | 换国内源: `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| GPU 检测不到 | `torch.cuda.is_available()` 检查 |

## 关联笔记
- [[Docker 与可复现环境]] — 更强的环境隔离
- [[HuggingFace工具链]]
