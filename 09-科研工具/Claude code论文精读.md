`evil-read-arxiv` 生成 π₀ 论文深度阅读笔记 —— 完整操作笔记**  
参考笔记
[Claude Code+Obsidian：每天自动读论文，早上到工位来上这么一篇真是惬意呀~_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1QfPezyEMz/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=1cc42850374daee933a9079575072bd5)
[如何让Claude code/Codex帮你精读论文+写笔记？_codex改论文-CSDN博客](https://blog.csdn.net/dahuoji0917/article/details/160033550)

---
## 1. 简介

`evil-read-arxiv` 是 Claude Code 的技能包，能实现： - 每天自动推荐具身智能相关论文 - 一键深度分析单篇论文（摘要、方法、实验、优缺点、图片提取） - 自动生成 Obsidian 笔记并提取高清图 - 支持顶会论文过滤

结合我们课题组的 `research_interests.yaml` 配置后，推荐结果会**高度聚焦 VLA、Grasp、Affordance** 三大方向。

---
## 2. 前置条件

- 已安装 **Claude Code** 并配置好模型 
- Git 已安装 
- Obsidian Vault 为 `E:\Obsidian\Evo-Matrix`(切换成自己的)

---

## 3. 安装与配置（一次性完成）

### 步骤1：克隆仓库（已完成）

```powershell 
cd E:\Obsidian git clone https://github.com/juliye2025/evil-read-arxiv.git
```
### 步骤2：在 Evo-Matrix 中创建必要文件夹
```powershell
cd E:\Obsidian\Evo-Matrix
mkdir -p 10_Daily 20_Research/Papers/{VLA,Affordance,Grasp,Other} 99_System/Config
```
### 步骤3：复制并配置 research_interests.yaml
```powershell
Copy-Item -Path "..\evil-read-arxiv\config.example.yaml" -Destination "99_System\Config\research_interests.yaml" -Force
code 99_System\Config\research_interests.yaml
```
使用以下具身智能专用配置（自己优化）：
```powershell
# Evo-Matrix 具身智能知识库专用配置（2026-05-10 精准版）
vault_path: "E:/Obsidian/Evo-Matrix"

research_domains:
  - name: "VLA_Embodied"
    arxiv_categories: ["cs.RO", "cs.AI", "cs.LG", "cs.CV"]
    keywords:
      - "Vision-Language-Action"
      - "Vision Language Action"
      - "VLA model"
      - "embodied VLA"
      - "robotic VLA"
      - "VLA for robotics"
      - "RT-2"
      - "PaLM-E"
      - "embodied vision-language-action"

  - name: "Grasp_Task"
    arxiv_categories: ["cs.RO", "cs.CV"]
    keywords:
      - "robotic grasping"
      - "dexterous grasping"
      - "task-oriented grasping"
      - "task-specific grasping"
      - "grasp synthesis"
      - "dexterous manipulation"
      - "in-hand manipulation"
      - "language-guided grasping"

  - name: "Affordance_Prediction"
    arxiv_categories: ["cs.RO", "cs.CV"]
    keywords:
      - "affordance prediction"
      - "affordance learning"
      - "visual affordance"
      - "3D affordance"
      - "affordance grounding"
      - "task-oriented affordance"
      - "embodied affordance"
      - "affordance field"
      - "Scene-MMKG"
      - "ManipMob-MMKG"

daily_recommend: 8
```
### 步骤4：复制技能到 Claude Code
```powershell
$claudeSkillsPath = "$env:USERPROFILE\.claude\skills"
Copy-Item -Path "..\evil-read-arxiv\start-my-day" -Destination $claudeSkillsPath -Recurse -Force
Copy-Item -Path "..\evil-read-arxiv\paper-analyze" -Destination $claudeSkillsPath -Recurse -Force
Copy-Item -Path "..\evil-read-arxiv\extract-paper-images" -Destination $claudeSkillsPath -Recurse -Force
Copy-Item -Path "..\evil-read-arxiv\paper-search" -Destination $claudeSkillsPath -Recurse -Force
```

## 4. 日常使用流程（推荐）

### 每日论文推荐（最常用）
```powershell
cd E:\Obsidian\Evo-Matrix
git pull origin master          # 先同步最新知识库
start-my-day                    # claudecode 自动抓取 + 分析 Top 论文
git add .
git commit -m "docs: 2026-05-XX 每日论文推荐"
git push origin master
```
### 单篇论文深度分析
```powershell
paper-analyze 2307.15818     # 示例：RT-2  论文编号
# 或
paper-analyze 2410.24164     # π₀
```
### 其他常用命令
```
- paper-search "VLA affordance" → 在已有笔记里搜索
- extract-paper-images <arxiv-id> → 仅提取图片
- conf-papers → 抓取顶会（CVPR/ICCV/ICLR 等）最新论文
```
## 5. 与 Evo-Matrix 知识库集成

1. 生成的笔记默认落在 20_Research/Papers/对应方向/
2. **手动建立链接**（推荐）：
- 在笔记顶部添加：[[02-VLA/📌-VLA总览]]
- 在对应 MOC（如 02-VLA/📌-VLA总览.md）中追加该笔记链接

## 常见问题 & 解决
|问题|解决办法|
|---|---|
|推荐论文不精准|修改 `research_interests.yaml` 中的 keywords 并重新运行|
|命令找不到|确保已复制技能到 `~/.claude/skills/` 并重启 Claude Code|
|路径错误|确认 `research_interests.yaml` 中的 `vault_path` 是 `E:/Obsidian/Evo-Matrix`（正斜杠）|
|生成笔记没有图片|手动运行 `extract-paper-images <id>`|
|Git 冲突|操作前必须 `git pull origin master`|

**注意**：
evil-read-arxiv 默认输出了「Obsidian 专属 Wikilink 语法」，而不是标准的 Markdown 图片语法
**生成效果**：
![[Pasted image 20260511113233.png]]


**操作完成！**  
运行 paper-analyze 2410.24164 后，论文笔记的具体存放位置如下：
最终保存路径
```JSON
E:\Obsidian\Evo-Matrix\
└── 20_Research\
    └── Papers\
        └── VLA\                          ← 因为这篇是 VLA 方向（π₀）
            └── 2410.24164-π₀-vision-language-action-flow-model.md   ← 自动生成的文件名
```
- **文件夹**：20_Research/Papers/VLA/（工具会根据 research_interests.yaml 中匹配的 VLA_Embodied 域自动选择）
- **文件名**：2410.24164- 开头 + 论文标题的英文 slug（标题可能有微小差异，但一定以 arXiv ID 开头）
- **图片位置**：笔记同目录下会自动创建一个 images/ 文件夹，里面存放所有提取的高清图片