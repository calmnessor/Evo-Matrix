# 计算机视觉与 ViT

> VLA 的 V = Vision——机器人通过视觉感知世界。理解 ViT 和视觉编码器，是理解 VLA 感知模块的前提。

## 为什么 VLA 需要理解视觉？

VLA 中，视觉负责回答：
- **"什么东西"**：桌子上有哪些物体？哪只是杯子？（分类/检测）
- **"哪里"**：物体在 3D 空间的什么位置？（定位/深度估计）
- **"什么状态"**：抽屉是开的还是关的？液体是否倒出来了？（状态识别）
- **"怎样操作"**：杯柄朝向哪边？从哪里抓？（位姿估计/可供性）

这些信息都要从一个或多个 RGB/深度摄像头图像中提取。

---

## 1. 范式转变：为什么 ViT 取代了 CNN？

### 1.1 CNN 的辉煌与局限

CNN 的核心是 **局部感受野 + 权重共享 + 层次化池化**。它天生带有两个**归纳偏置 (Inductive Bias)**：

1. **局部性**：特征靠邻近像素拼接——需要堆很多层才能看到全局
2. **平移等变性**：同样的模式在不同位置激活相同的滤波器

**这对 VLA 意味着什么？**
- CNN 擅长"这个像素区域是不是杯子边缘"（局部纹理）
- CNN 不擅长"杯子和桌面的空间关系"（需要全局推理）
- 在机器人场景中，物体经常出现在奇怪的位置/角度 → CNN 的平移等变性不足以应对旋转和 3D 变换

### 1.2 ViT 的革命：扔掉卷积，全靠 Attention

ViT 把图像切成 patch，每个 patch 当作一个"单词"喂给 Transformer。

**对比表**：

| | CNN (ResNet-50) | ViT (ViT-B/16) |
|------|------|------|
| 感受野 | 局部 → 逐层扩大 | **天然全局** (第一层就能看到整个图) |
| 归纳偏置 | 强（局部、平移等变） | 弱（数据驱动） |
| 数据需求 | 中等 (ImageNet 1M) | 大 (JFT 300M 或 ImageNet-21K) |
| 参数量 | 25M | 86M |
| 推理速度 | 快 | 慢（attention $O(N^2)$, N=patch 数） |
| VLA 中 | 已被替代 | **主流** |

### 1.3 什么时候 CNN 仍然有用？

- **轻量任务**：简单的物体检测、色块定位（轻量 CNN 比 ViT 快得多）
- **时序压缩**：RT-1 中用了 EfficientNet 提取每帧特征
- **嵌入式部署**：ViT 的 attention 开销在树莓派等边缘设备上不划算

---

## 2. ViT 架构深度拆解

```
输入: 图像 [3, 224, 224]

步骤 1: Patch Embedding
  → 切成 14×14=196 个 16×16 patch
  → 每个 patch 展平为 16×16×3=768 维向量
  → 线性投影 [768] → [d_model] (如 768→768 or 768→1024)

步骤 2: 加 CLS Token + Position Embedding
  → [CLS] token (可学习, 尺寸 [1, d_model]) — BERT 风格，用于分类
  → Position Embedding [197, d_model] — 可学习或正余弦

步骤 3: Transformer Encoder × N 层
  每层: MSA + FFN + Residual + LayerNorm
  输出: [197, d_model]  (CLS token + 196 patch tokens)

步骤 4: 根据任务取不同输出
  分类: 取 CLS token → MLP Head → 类别
  分割: 取所有 patch token → reshape → 上采样
  VLA: 取所有 patch token → 投影 → 喂给 LLM
```

**Patch Size 的权衡**：
- 小 patch (14×14)：更多 token (196)，细节丰富，计算重
- 大 patch (32×32)：更少 token (49)，速度快，细节损失

SigLIP 用 patch=14, 384×384 图像 → 产生 (384/14)² = 729 个 token。

---

## 3. 从 2D 到 3D：视觉的更深层

### 3.1 深度估计

**单目深度估计**：从单张 RGB 图预测每个像素的深度。

| 方法 | 原理 | 特点 |
|------|------|------|
| **ZoeDepth** | 相对深度 (metric-aware) | 适合室内桌面操作 |
| **Depth Anything v2** | 大规模伪标注预训练 | 鲁棒性极强 |
| **Metric3D** | 绝对深度，跨相机泛化 | 直接输出米制深度 |

> VLA 中可以加入深度估计分支，从单 RGB 获取 3D 信息。

### 3.2 目标检测与分割

- **DETR**：ViT 时代的检测器，用 Transformer 直接预测物体集合
- **SAM (Segment Anything)**：零样本实例分割
- **SAM-3D**：把 SAM 延伸到 3D 点云

---

## 4. 视觉在 VLA 中的典型流程

```
实际部署链路:

RealSense 深度相机
    ↓
RGB 图像 [480, 640, 3] + 深度图 [480, 640]
    ↓
Resize/Crop → [384, 384, 3]
    ↓
SigLIP ViT → visual tokens [729, 1152]
    ↓
2 层 MLP 投影器 → visual tokens [729, 4096]
    ↓                                   ↓
与语言 token 拼接       深度图 → 点云 → 3D 特征
    ↓                                   ↓
LLaMA Decoder ←──────── 融合特征 ────────┘
    ↓
动作 token (离散 bin)
    ↓
解码为连续动作 [Δx, Δy, Δz, Δr, Δp, Δy, gripper]
```

---

## 5. 动手：用 SigLIP 提取视觉特征

```python
import torch
from transformers import AutoModel, AutoProcessor

# 加载 SigLIP (OpenVLA 同款)
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# 模拟一张机器人视角的图像
from PIL import Image
image = Image.open("robot_view.jpg")

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model.vision_model(**inputs)

# last_hidden_state: [1, 729, 1152]
# 729 = (384/14)^2 个 patch token
print(f"Visual tokens shape: {outputs.last_hidden_state.shape}")

# 如果需要，投影到 LLM hidden dim
projector = torch.nn.Sequential(
    torch.nn.Linear(1152, 4096),
    torch.nn.GELU(),
    torch.nn.Linear(4096, 4096),
)
visual_features = projector(outputs.last_hidden_state)
print(f"Projected tokens: {visual_features.shape}")  # [1, 729, 4096]
```

---

## 6. 新手学习路线

### 第一阶段：从 CNN 到 ViT (1-2 天)
1. 用 ResNet-18 跑一次 ImageNet 的简单分类
2. 用 ViT-B/16 跑同样的分类，对比参数量和计算量
3. 可视化 ViT 每一层的 attention map ← 理解"模型在看哪里"

### 第二阶段：理解 CLIP/SigLIP (2-3 天)
1. 读 CLIP 论文，理解对比学习的目标
2. 用 CLIP 做零样本分类：不训练，直接推理你的图片
3. 理解 SigLIP 的 sigmoid loss 和 CLIP softmax loss 的区别

### 第三阶段：VLA 视觉 (3-4 天)
1. 读 OpenVLA 论文/代码中视觉编码器的部分
2. 用上面代码提取 SigLIP 特征，投影到 4096 维
3. 设想：如果加入深度估计分支，visual token 应该如何扩展？

---

## 7. 推荐论文与资源

| # | 论文 | 关键词 | 理由 |
|---|------|--------|------|
| 1 | **ViT** (2010.11929) | Patch + Transformer | 视觉 Transformer 开山之作 |
| 2 | **CLIP** (2103.00020) | 对比学习 | VLA 视觉编码器的理论基础 |
| 3 | **SigLIP** (2303.15343) | Sigmoid Loss | OpenVLA 选用的编码器 |
| 4 | **DINOv2** (2304.07193) | 自监督视觉 | 更强空间理解的潜质 |
| 5 | **RT-2** (2307.15818) | VLA 视觉-语言 | 看 ViT 在 VLA 中怎么用的 |

---

## 8. 自检问题

### 基础关
- [ ] 我能画出 ViT 的完整数据流（patch → embedding → Transformer → 输出）
- [ ] 我理解 Patch Embedding 的原理和为什么需要 Position Embedding
- [ ] 我知道 CNN 和 ViT 的核心区别（局部 vs 全局感受野）
- [ ] 我知道 CLIP 的对比学习训练方式

### 进阶关
- [ ] 我知道 SigLIP 相对于 CLIP 的改进（sigmoid loss 替代 softmax）
- [ ] 我能解释视觉 token 的数量如何计算（(H/patch) × (W/patch)）
- [ ] 我理解 DINOv2 和 CLIP/SigLIP 在预训练目标上的差异
- [ ] 我能说出 OpenVLA 视觉编码器的完整配置

### 实战关
- [ ] 我能用 SigLIP 提取图像特征
- [ ] 我能理解 OpenVLA 源码中视觉投影器的定义
- [ ] 我知道在 VLA 中 visual token 如何与 text token 拼接
- [ ] 我能评估不同视觉编码器对 VLA 性能的影响

---

## 关联笔记

- [[CLIP与多模态对齐]] — 对比学习 + SigLIP vs CLIP + OpenVLA 配置
- [[DINOv2与自监督视觉]] — 自监督视觉特征 + 3D 几何理解
- [[../Transformer与注意力/Transformer与注意力|Transformer与注意力]] — ViT 的底层引擎
- [[../3D视觉/3D视觉与点云|3D视觉与点云]] — 从 2D 视觉扩展到 3D 感知
- [[../../02-VLA/VLA模型总览|VLA模型总览]] — 视觉编码器在 VLA 中的位置和角色
