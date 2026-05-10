# Obsidian 进阶技巧

> 让 Obsidian 从记事本变成知识引擎

## 核心功能

### 双向链接
```markdown
[[文件名]]           → 链接到笔记
[[文件名|显示文本]]   → 别名链接
[[文件名#标题]]      → 链接到章节
[[文件名^block]]    → 链接到段落
```

### 标签
在 frontmatter 中定义:
```yaml
---
tags: [VLA, 论文, 精读]
---
```

### 关系图谱
- 打开图谱面板 (左侧点 Graph View)
- 局部图谱 (Local Graph): 只看当前笔记的关联
- 按文件夹/标签上颜色

## 推荐插件

| 插件 | 用途 | 优先级 |
|------|------|--------|
| **Dataview** | 用查询语法生成动态列表 | ⭐⭐⭐ |
| **Templater** | 高级模板（自动日期、文件移动） | ⭐⭐⭐ |
| **Excalidraw** | 白板画图 | ⭐⭐ |
| **Paste URL** | 粘贴 URL 自动生成标题 | ⭐⭐ |
| **Zotero Integration** | 直接导入 Zotero 论文 | ⭐⭐⭐ |
| **Quick Add** | 快速创建笔记 | ⭐⭐ |

## Dataview 查询示例

### 列出所有待读论文
```dataview
TABLE year, tags
FROM "02-VLA" OR "03-Affordance" OR "04-Grasp"
WHERE contains(tags, "待读")
SORT year DESC
```

### 列出最近修改的笔记
```dataview
TABLE file.mtime as "修改时间"
FROM ""
SORT file.mtime DESC
LIMIT 10
```

### 列出所有实验任务
```dataview
TASK
FROM "02-VLA" OR "03-Affordance" OR "04-Grasp"
WHERE !completed
```

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+O` | 快速打开文件 |
| `Ctrl+P` | 命令面板 |
| `Ctrl+E` | 切换编辑/预览 |
| `Ctrl+K` → `Ctrl+K` | 创建链接 |
| `Ctrl+N` | 新建笔记 |
| `Ctrl+Shift+F` | 全文搜索 |

## 课题组协作建议

如果多人共用一个 Vault:
1. Obsidian Sync 或把 vault 放在共享盘
2. 用 Git 同步 `.md` 文件（注意 `.obsidian/` 配置冲突）
3. 每个人在 YAML frontmatter 注明作者

## 关联笔记
- [[🏠-总览]] — 知识库入口
- [[🗺️-学习路线]] — 结构化的学习路径
