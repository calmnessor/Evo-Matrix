# Git 与版本控制

> 代码的"时间机器"——实验记录、协作和备份都离不开

## 基础工作流

```bash
# 初始化
git init
git add .
git commit -m "init"

# 分支
git checkout -b exp/diffusion-lr1e-4
# ... 做实验 ...
git add -A
git commit -m "exp: diffusion policy lr=1e-4, success 78%"
```

## 科研项目建议

### Commit 规范
使用语义化 commit message：
```
exp: <实验描述> — <关键结果>
fix: <问题修复>
feat: <新增功能>
refactor: <重构>
docs: <文档>
```

### 分支策略
```
main           ← 稳定的代码
├── exp/lr-sweep-v1    ← 学习率扫描
├── exp/new-action-tokenizer  ← 新 tokenizer
└── paper/final-experiments   ← 定稿实验
```

### .gitignore（重要！）
```gitignore
# Python
__pycache__/
*.pyc
*.egg-info/

# 环境
env/
venv/
.conda/

# 数据（太大了，用 DVC 管理）
data/
*.h5
*.tfrecord

# 模型权重
checkpoints/
*.pt
*.pth

# IDE
.vscode/
.idea/

# 密钥
.env
*.key

# 输出
logs/
wandb/
```

## 常见协作场景

### Clone 仓库 + 安装
```bash
git clone https://github.com/openvla/openvla
cd openvla
pip install -e .
```

### 拉取最新代码
```bash
git pull origin main
```

### 从 GitHub 上给开源项目提 PR
1. Fork → Clone
2. 在新分支修改
3. Push → 创建 Pull Request
4. 等待审查和讨论
5. Merge！

## 踩坑记录

| 问题 | 解决 |
|------|------|
| 提交了敏感文件 | `git rm --cached <file>` + 加到 `.gitignore` |
| 改坏了想回退 | `git checkout -- <file>` |
| 忘了在哪个分支 | `git branch -a` |
| commit 信息写错了 | `git commit --amend` (仅限未 push) |
| 同步 fork 仓库 | `git remote add upstream <url>` + `git fetch upstream` |

## 关联笔记
- [[Claude Code 使用指南]] — Claude Code 可帮你解决 git 冲突
- [[Python 环境管理]] — 代码版本 + 环境版本一起管理
- [[00-入口/🧪-实验总索引|实验索引]]
