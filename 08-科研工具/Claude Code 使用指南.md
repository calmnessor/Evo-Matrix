# Claude Code 使用指南

> Claude Code 是 Anthropic 出品的 AI 编程助手，可以在终端/IDE 中直接对话编程

## 安装与配置

### 安装
```bash
# 通过 npm 安装
npm install -g @anthropic-ai/claude-code

# 启动
claude
```

### 配置
```bash
# 设置模型
claude config set model claude-opus-4-7

# 设置主题
claude config set theme dark

# 查看配置
claude config list
```

## 常用命令

| 命令 | 作用 |
|------|------|
| `/help` | 帮助 |
| `/config` | 配置 |
| `/clear` | 清空对话 |
| `/init` | 初始化项目 CLAUDE.md |
| `/review` | PR 审查 |
| `/simplify` | 代码重构优化 |

## 核心使用场景

### 1. 代码理解
```
"解释 OpenVLA 的 forward 函数"
"这个类的调用链是什么"
"找出所有修改 action space 的地方"
```

### 2. 代码生成
```
"写一个 PyTorch 数据加载器，支持 OXE RLDS 格式"
"帮我写一个计算正运动学的函数"
```

### 3. 调试
```
"这段代码报 IndexError，帮我看看为什么"
"梯度为 None，排查一下"
```

### 4. 重构
```
"把这段代码从 for 循环改成向量化"
"用 einops 重写张量操作"
```

### 5. 论文复现
```
"根据这张论文架构图，写一个 PyTorch 实现"
"帮我对比我的实现和论文的伪代码"
```

## 课题组常用 Prompt

### 理解论文代码
```
阅读这个仓库的代码结构，解释：
1. 数据流是怎样的
2. 训练循环的核心逻辑
3. 关键配置项在哪里
```

### 复现论文模块
```
根据以下论文描述，实现 X 模块：
[贴入论文相关段落]
要求:
- PyTorch 实现
- 包含 forward
- 写一个简单的 test case
```

### 实验排查
```
我的训练 loss 不下降了，帮我排查:
1. [贴入训练曲线描述]
2. [贴入关键代码]
3. [贴入超参]
```

## 技巧

### CLAUDE.md 文件
在项目根目录放 `CLAUDE.md` 文件，描述项目背景、架构、约定。Claude Code 每次启动会自动读取，大幅提高效率。

```markdown
# PROJECT: OpenVLA Reproduction
- 目标: 复现 OpenVLA 在 Bridge v2 上的结果
- 框架: PyTorch + HuggingFace Transformers
- 数据: RLDS 格式
- 关键文件:
  - model.py: VLA 模型定义
  - data.py: 数据加载
  - train.py: LoRA 微调脚本
```

### 善用 /init
在项目根目录跑 `/init`，Claude Code 自动分析代码结构生成 CLAUDE.md。

### 多文件操作
Claude Code 可以同时编辑多个文件、运行命令、执行测试。大胆提复杂需求：

```
"重构 model.py: 把 vision encoder 抽成独立模块，并更新 train.py 的导入路径，然后运行 pytest"
```

## 注意事项

- Claude Code 有上下文限制，长对话可以 `/clear` 重建
- 生成的代码需要人工审查，不要直接粘贴到生产环境
- 不要粘贴敏感信息（密钥、内部数据路径等）
- 论文复现的代码先在小数据集验证再扩展

## 关联笔记
- [[AI辅助论文写作]] — 论文写作中的 AI 使用
- [[Git 与版本控制]] — 与 Claude Code 协作的最佳实践
