```markdown
# Git 与版本控制 —— Evo-Matrix 课题组知识库协作指南

> 知识库的“时间机器”——论文笔记、思路演进、实验记录、团队协作都离不开 Git

**适用对象**：Evo-Matrix（Obsidian + Markdown）知识库所有组员  
**当前分支**：`master`（请严格按照本指南操作）

---

## 1. 推荐工具（新手最友好）

### Obsidian Git 插件（强烈推荐！）
- 在 Obsidian 界面内直接 **Pull / Commit / Push**，无需打开终端
- 安装方法：Obsidian → 设置 → 社区插件 → 搜索 “Obsidian Git” → 安装并启用
- 常用快捷键（可在插件设置中自定义）：
  - `Ctrl + Shift + P` → `Git: Pull`（拉取最新代码）
  - `Git: Commit`（提交修改）
  - `Git: Push`（推送到远程）

**新手建议**：日常 90% 的操作都用 **Obsidian Git 插件** 完成，只有复杂操作才打开 PowerShell。

---

## 2. Evo-Matrix 知识库专用工作流（最常用）

### 每日标准流程（跑完 `start-my-day` 后）
```bash
cd E:\Obsidian\Evo-Matrix

# 1. 必须先拉取最新内容（防止冲突）
git pull origin master

# 2. 运行自动抓论文（可选）
start-my-day

# 3. 修改笔记、建立链接、更新 MOC
# 4. 提交并推送
git add .
git commit -m "docs: 2026-05-10 每日论文推荐 + 新增 VLA 笔记"
git push origin master
```

**使用 Obsidian Git 插件时**：
1. 修改完笔记
2. 按 `Ctrl + Shift + P` → 输入 `Git: Commit`
3. 输入 commit 信息 → 提交
4. 再执行 `Git: Push`

---

## 3. Commit 规范（请严格遵守）

使用语义化信息，方便全组和导师快速了解改动：

| 类型       | 示例                                      | 说明                     |
|------------|-------------------------------------------|--------------------------|
| `docs`     | `docs: 新增 RT-2 VLA 论文笔记`           | 文档、论文笔记           |
| `feat`     | `feat: 完善 Affordance MOC + 知识链接`   | 新增功能/模块            |
| `refactor` | `refactor: 优化 02-VLA 目录结构`         | 重构                     |
| `fix`      | `fix: 修复 [[链接]] 失效问题`            | 修复错误                 |
| `chore`    | `chore: 更新 research_interests.yaml`    | 配置、工具更新           |

**推荐好例子**：
- `docs: 新增 Grasp 方向 DemoGrasp 论文 + 关联 Affordance 笔记`
- `feat: 张三-思路演进.canvas 第 19 周更新`

---

## 4. 分支策略（当前仓库使用 master）

```
master                  ← 稳定主分支（最终合并到这里）
├── feature/vla-new-model      ← 新增 VLA 模型笔记
├── feature/affordance-moc     ← 完善 Affordance 总览
├── bugfix/broken-link         ← 修复链接错误
└── personal/张三-周报         ← 个人每周进展（可选）
```

**建议**：
- 小修改（新增一篇论文笔记）可直接推 `master`
- 重要改动（新增目录、修改模板、大规模重构）请新建分支 + 提交 PR

---

## 5. .gitignore（已优化，建议直接使用）

```gitignore
# Obsidian 临时文件
.obsidian/
.trash/
.cache/

# Python / 虚拟环境
__pycache__/
*.pyc
venv/
.venv/
.env

# 大文件（知识库不需要提交）
*.pdf
*.h5
*.pt
*.pth
checkpoints/
data/

# 日志和临时文件
logs/
wandb/
*.log

# 密钥（重要！）
api_settings.json
*.key
```

---

## 6. 常见协作场景

### 场景1：日常更新知识库（最常用）
1. 打开 Obsidian → `Git: Pull`
2. 新增/修改笔记、建立 [[双向链接]]
3. `Git: Commit` + `Git: Push`

### 场景2：提交 Pull Request（安全推荐）
1. `git checkout -b feature/你的功能名`
2. 修改文件
3. `git add . && git commit -m "..." `
4. `git push origin feature/你的功能名`
5. 去 GitHub 网页创建 Pull Request → 等待审核

### 场景3：同步最新代码（别人提交了内容）
```bash
git pull origin master
```

---

## 7. 踩坑记录 & 解决方案

| 问题                          | 解决办法                                      |
|-------------------------------|-----------------------------------------------|
| 别人改了同一篇笔记导致冲突    | 先 `git pull` 再 Commit 自己的修改           |
| [[链接]] 失效                 | 在 Obsidian Graph View 检查并修复             |
| 提交了 `.obsidian` 文件夹     | 加到 .gitignore 并执行 `git rm --cached .obsidian` |
| 想回退某个文件                | `git checkout -- 文件名` 或 Obsidian 版本历史 |
| 不知道当前分支                | `git branch` 或 Obsidian Git 状态栏          |
| Commit 信息写错               | `git commit --amend`（未 push 前有效）       |

---

## 8. 关联笔记

- [[Claude Code 使用指南]] —— 自动抓论文 + 生成笔记
- [[06-科研工具/evil-read-arxiv 配置]] —— 每天论文推荐
- [[07-组员研究进展/全组研究仪表盘]] —— 组员思路可视化
- [[00-入口/🧪-实验总索引]] —— 实验记录索引

---

**最后提醒**：
- **每次操作前必须先 Pull**（`git pull origin master`）
- 重要改动建议走 **PR**（已开启分支保护）
- 有任何 Git 问题，随时在群里提问或在仓库 Issues 中提交

**欢迎大家一起把 Evo-Matrix 建设成课题组最强科研大脑！** 🚀
```

---

**使用方法**：
1. 直接复制上面**全部内容**
2. 粘贴覆盖你的 `README.md`（或新建 `Git-协作指南.md` 放在 `06-科研工具/` 目录）
3. 用之前的命令推送：
   ```powershell
   cd E:\Obsidian\Evo-Matrix
   git add README.md
   git commit -m "docs: 完善 Git 使用指南（Evo-Matrix 知识库专用）"
   git push origin master
   ```

需要我再调整某个部分（加图片说明、简化某些内容等）随时告诉我！