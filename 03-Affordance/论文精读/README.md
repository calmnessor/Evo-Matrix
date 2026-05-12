# Affordance 论文精读

> 按技术路线组织的 Affordance 论文笔记，覆盖 2D 基础模型、3DGS 表征、关键点约束三条路线。

## 已组织论文

### 2D Affordance（基础模型驱动）

从 2D 视觉基础模型（DINOv2、VLM）中蒸馏 affordance 知识，无需 3D 标注。

| # | 论文 | 日期 | 核心贡献 |
|---|------|------|----------|
| 1 | [[UAD]] | 2025.06 | 无监督 affordance 蒸馏：DINOv2 + GPT-4o → 零样本泛化 |

### 3D Affordance（3DGS 表征）

以 3D Gaussian Splatting 为 affordance 表征，从单物体到场景级序列推理。

| # | 论文 | 日期 | 核心贡献 |
|---|------|------|----------|
| 1 | [[3DAffordSplat]] | 2025.04 | 首个 3DGS affordance 数据集 + AffordSplatNet（CVPR 2025） |
| 2 | [[SeqAffordSplat]] | 2025.07 | 场景级序列化 affordance 推理，LLM 规划 + 3DGS 执行（AAAI 2026） |

### 关键点约束（结构化 Affordance）

将 affordance 表示为关系关键点约束，通过优化求解器生成动作。

| # | 论文 | 日期 | 核心贡献 |
|---|------|------|----------|
| 1 | [[ReKep]] | 2024.09 | 关系关键点约束 → 结构化 affordance 表示（CoRL 2024） |

## 待组织论文

以下核心论文尚未纳入精读体系：

| 论文 | 年份 | 归类 | 值得读的原因 |
|------|------|------|-------------|
| Where2Act | 2021 | 2D Affordance | 像素级 affordance 开创性工作 |
| ManiGaussian | 2024 | 3D Affordance | 3DGS + 动态 affordance 先驱 |
| GAPartNet | 2023 | 结构化 | 物体部件级 affordance 数据集 |
| GauTOAO | 2024 | 3D Affordance | 3DGS 任务导向 affordance |
| Robo-ABC | 2024 | 2D Affordance | 大规模 affordance 数据自动生成 |
| AnyGrasp | 2023 | Grasp → Affordance | 7-DoF 通用抓取，实时性极好 |

---

👉 新建论文笔记: 使用 [[../../templates/论文笔记模板|论文笔记模板]]
