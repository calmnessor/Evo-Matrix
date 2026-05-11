# Evo-Matrix — 具身智能课题组知识库

> 一个基于 [Obsidian](https://obsidian.md) 的结构化知识库，服务于具身智能三个研究方向（VLA / Affordance 预测 / Grasp）的课题组协作学习与科研追踪。

<p align="center">
  <img src="https://img.shields.io/badge/具身智能-Embodied_AI-2563EB?style=for-the-badge" />
  <img src="https://img.shields.io/badge/VLA-Vision_Language_Action-F59E0B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Grasp-Robotic_Grasping-16A34A?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Affordance-Affordance_Prediction-8B5CF6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Sim2Real-Sim_to_Real-0D9488?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Manipulation-Dexterous_Manipulation-DC2626?style=for-the-badge" />
  <img src="https://img.shields.io/badge/RL-Robot_Learning-7C3AED?style=for-the-badge" />
</p>

## 这是什么？

Evo-Matrix 是课题组共享的"第二大脑"——用双向链接和知识图谱组织具身智能领域的理论知识、论文笔记、实验记录和科研工具。新人可以按学习路线图快速入门，进行中课题的研究员可以追踪论文、记录实验、建立知识之间的交叉链接。

## 三大研究方向

| 方向 | 简介 | 入口 |
|------|------|------|
| 🤖 **VLA** | Vision-Language-Action：端到端感知-决策-控制 | [`02-VLA/📌-VLA总览.md`](02-VLA/📌-VLA总览.md) |
| 🎯 **Affordance 预测** | 理解物体"能用来做什么"——动作可能性 | [`03-Affordance/📌-Affordance总览.md`](03-Affordance/📌-Affordance总览.md) |
| ✋ **Grasp** | 让机器人"拿得起、握得稳" | [`04-Grasp/📌-Grasp总览.md`](04-Grasp/📌-Grasp总览.md) |

## 快速开始

```bash
git clone https://github.com/calmnessor/Evo-Matrix.git
```

用 [Obsidian](https://obsidian.md) 打开克隆的文件夹 → "Open folder as vault"。

入门路径：
1. 从 [`00-入口/🏠-总览.md`](00-入口/🏠-总览.md) 了解全局结构
2. 新成员读 [`00-入口/🚀-新人指南.md`](00-入口/🚀-新人指南.md) 搭建环境
3. 按 [`00-入口/🗺️-学习路线.md`](00-入口/🗺️-学习路线.md) 开始 14 天速通 + 30 天深入学习

## 知识库结构

```
Evo-Matrix/
├── 00-入口/               ← 全局 MOC、新人指南、学习路线、论文/实验总索引
├── 01-基础理论/            ← 深度学习、Transformer、CV、LLM、运动学、RL、3D 视觉（三方向共享）
├── 02-VLA/                ← VLA 方向：模型架构、论文精读、模型复现、实验项目、成员笔记
├── 03-Affordance/         ← Affordance 方向：方法理论、论文精读、实验项目、成员笔记
├── 04-Grasp/              ← Grasp 方向：抓取方法、论文精读、实验项目、成员笔记
├── 05-仿真与工具/          ← MuJoCo、HuggingFace、Blender、数据集索引
├── 06-科研工具/            ← Claude Code、LaTeX、Git、Python 环境、文献检索
└── templates/             ← 论文笔记、实验记录、费曼笔记模板
```

## 每个研究方向包含

```
0X-方向/
├── 📌-总览.md          ← 方向 MOC：技术栈、模型谱系、关键论文
├── 论文精读/           ← 成员论文笔记（使用统一模板）
├── 模型复现/           ← 代表性模型复现记录（环境 + 训练 + 踩坑）
├── 实验项目/           ← 实验记录与项目追踪
└── 成员笔记/           ← 个人学习笔记与费曼输出
```

## 核心特性

- **三方向并行**: VLA / Affordance / Grasp 各有独立空间，通过双向链接交叉关联
- **MOC（内容地图）**: 每个目录的索引页作为该方向的导航入口
- **双向链接**: 知识节点之间通过 `[[wiki-link]]` 互相连接，形成可跳转的知识图谱
- **学习路线**: 14 天速通 + 30 天深入，从零基础到能独立部署和微调 VLA 模型
- **费曼学习法**: 每个知识点都要能"用自己的话讲出来"，模板内置费曼检查清单
- **论文追踪**: 统一的论文笔记模板，按方向组织，通过标签和链接关联到相关方法/实验
- **实验可复现**: 实验记录模板包含环境配置、关键指标、踩坑记录

## 协作方式

- **新人**: Fork → 在本地按学习路线推进 → 用费曼模板写笔记 → PR 到主仓库
- **课题组成员**: 在对应方向的 `论文精读/` 或 `实验项目/` 下新建笔记 → 定期 PR review
- **外部贡献**: 欢迎通过 Issue/PR 补充论文、修正错误、添加工具教程

## 推荐 Obsidian 插件

| 插件 | 用途 |
|------|------|
| Dataview | 动态列表（如列出所有待读论文） |
| Templater | 高级模板自动插入 |
| Excalidraw | 画架构图/思维导图 |
| Zotero Integration | 从 Zotero 直接导入论文 |

## 贡献者

<a href="https://github.com/calmnessor/Evo-Matrix/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=calmnessor/Evo-Matrix" />
</a>

> 只要向本仓库提交过 Commit，你的头像就会自动出现在上方贡献者墙中。

## 项目统计

![visitors](https://komarev.com/ghpvc/?username=calmnessor&repo=Evo-Matrix&color=79C83D&style=flat)

[![Star History Chart](https://api.star-history.com/svg?repos=calmnessor/Evo-Matrix&type=Date)](https://star-history.com/#calmnessor/Evo-Matrix&Date)

## 许可

本知识库内容仅供课题组内部及学术用途。论文笔记和知识整理遵循原始论文的许可。

---

> 知识在连接中涌现——每个 `[[链接]]` 都是一次发现的可能。
