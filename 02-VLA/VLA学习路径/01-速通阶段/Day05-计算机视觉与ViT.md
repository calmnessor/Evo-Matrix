# 👁️ Day 5: 计算机视觉与 ViT（8 小时）

> **口号**: "让模型'看'懂世界——从像素到语义！"  
> **目标**: 理解 ViT 架构和 CLIP 等多模态模型，能使用 HuggingFace 视觉模型  
> **为什么重要**: VLA 的 V = Vision，机器人要先"看到"才能"行动"

---

## 📋 今日打卡清单

| 序号 | 任务 | 预计时间 | ✅ |
|------|------|----------|-----|
| 1 | 从 CNN 到 ViT 的范式转变 | 1h | [ ] |
| 2 | ViT 架构逐层拆解 | 1.5h | [ ] |
| 3 | CLIP 与多模态对齐 | 1.5h | [ ] |
| 4 | SigLIP 与 VLA 中的视觉编码器 | 1h | [ ] |
| 5 | HuggingFace 视觉模型实战 | 1.5h | [ ] |
| 6 | 费曼挑战 | 1.5h | [ ] |

---

## 模块 1: 从 CNN 到 ViT 的范式转变（1 小时）

### 1.1 CNN 的局限 — 20min

```
CNN 的问题:
  1. 局部感受野 → 需要很多层才能看到全局
  2. 固定的卷积核 → 不够灵活
  3. 空间归纳偏置 → 你假定"相邻像素相关"
     但对某些任务，全局理解更重要

就像看一张图找"所有红色的物体":
  CNN: 用小窗口一片一片扫（慢，容易漏）
  ViT: 一眼看全图，直接注意到所有红色区域（快，全面）
```

### 1.2 ViT 的核心思想 — 30min

```
ViT = "把图像当成句子处理"

图像 → 切成小 patch → 每个 patch 当做一个 "词" → 送入 Transformer

步骤:
  1. 输入图像 [3, 224, 224]
  2. 切成 14×14 个 patch，每个 16×16 像素
  3. 每个 patch 拉平成 768 维向量 (16×16×3)
  4. 加位置编码 + class token
  5. 送入标准 Transformer
  6. 输出用于分类/检测/...
```

```python
# 直观理解
image = torch.randn(3, 224, 224)  # H=224, W=224, C=3

# Patchify: 把图像切成 N=196 个 patch
patch_size = 16
n_patches = (224 // 16) * (224 // 16)  # 196
patches = image.unfold(1, 16, 16).unfold(2, 16, 16)
patches = patches.permute(1, 2, 0, 3, 4).reshape(n_patches, -1)
# patches shape: [196, 768]

# 然后就像处理文本 token 一样处理
```

### 1.3 CNN vs ViT 对比表 — 10min

| 特性 | CNN | ViT |
|------|-----|-----|
| 感受野 | 局部 → 逐渐全局 | 天然全局 |
| 归纳偏置 | 强（局部、平移不变） | 弱（数据驱动） |
| 数据需求 | 中等 | 大规模 |
| 计算效率 | 高 | 中等（大图时注意力开销大） |
| VLA 采用情况 | 逐渐被替代 | 主流选择 |

---

## 模块 2: ViT 架构逐层拆解（1.5 小时）

### 2.1 完整 ViT 代码实现 — 60min

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    """
    完整 ViT，与 OpenVLA 中使用的视觉编码器同架构
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        n_channels=3,
        d_model=768,
        n_heads=12,
        n_layers=12,
        ffn_dim=3072,
        n_classes=1000,
    ):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size

        # 步骤 1: Patch Embedding
        # 用一个 Conv2d 替代手动 slice（效果相同，更快）
        self.patch_embed = nn.Conv2d(
            n_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )

        # 步骤 2: Class Token + Position Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model))

        # 步骤 3: Transformer Blocks（和 Day 3 一模一样！）
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])

        # 步骤 4: 输出
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B = x.shape[0]

        # 1. Patch Embedding
        x = self.patch_embed(x)        # [B, d_model, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, d_model]

        # 2. Prepend Class Token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)          # [B, 1+n_patches, d_model]

        # 3. Add Position Embedding
        x = x + self.pos_embed

        # 4. Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 5. 取 CLS token 的输出做分类
        x = self.norm(x[:, 0])   # [B, d_model]
        return self.head(x)      # [B, n_classes]


# ====== 同样的 TransformerBlock（Day 3 的代码复用）======

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

### 2.2 关键设计决策 — 20min

```
问题 1: 为什么需要 CLS Token？
  → 它是一个"汇总"token，收集全图信息
  → 就像一篇文章的"总结句"
  → VLA 中可能不用 CLS，而是取所有 patch 的输出

问题 2: 为什么用 Conv2d 做 Patch Embedding？
  → Conv2d(kernel=16, stride=16) 等价于切 patch + 线性投影
  → 但计算效率更高（GPU 加速）

问题 3: 位置编码是必需的吗？
  → 是！没有位置编码，ViT 无法知道 patch 的空间位置
  → 想象把拼图打乱后还知道每个拼图块在哪——必须有位置信息
```

### 2.3 ViT 的变体 — 10min

| 变体 | 特点 | VLA 中 |
|------|------|--------|
| ViT-B/16 | Base 大小，patch=16 | 常用 |
| ViT-L/14 | Large 大小，patch=14 | OpenVLA 用这个 |
| SigLIP ViT | 用 Sigmoid 替代 Softmax 的对比学习 | **VLA 最新标配** |
| DINOv2 | 自监督训练，特征更好 | 一些 VLA 实验用 |

---

## 模块 3: CLIP 与多模态对齐（1.5 小时）

> CLIP 是 VLA 中视觉-语言对齐的核心。OpenVLA 就是用 CLIP/SigLIP 作为视觉编码器！

### 3.1 CLIP 的训练方式 — 30min

```
传统图像分类:
  图像 → CNN → "这是猫"

CLIP 的训练（对比学习）:
  批次内 N 对 (图像, 文本)
  
  图像编码器: img₁ → feat_img₁
  文本编码器: text₁ → feat_text₁
  
  计算 N×N 相似度矩阵:
           text₁  text₂  ...  text_N
    img₁ [  0.9    0.1   ...   0.0  ]  ← 对角线应该最大
    img₂ [  0.1    0.8   ...   0.1  ]
    ...
    img_N [  0.0    0.1   ...   0.9  ]
  
  目标: 让配对的 (img_i, text_i) 相似度最大，其他最小
```

### 3.2 CLIP 代码实现 — 30min

```python
class CLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.vision_encoder = vision_encoder   # ViT
        self.text_encoder = text_encoder       # Transformer

        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, images, texts):
        # 编码
        img_feat = self.vision_proj(self.vision_encoder(images))  # [B, D]
        txt_feat = self.text_proj(self.text_encoder(texts))        # [B, D]

        # L2 归一化
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        # 相似度矩阵
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feat @ txt_feat.T  # [B, B]

        # 对称损失
        labels = torch.arange(B, device=images.device)
        loss_img = F.cross_entropy(logits, labels)    # 图像方向
        loss_txt = F.cross_entropy(logits.T, labels)  # 文本方向
        return (loss_img + loss_txt) / 2
```

### 3.3 CLIP 在 VLA 中的角色 — 20min

```
VLA 推理时:
  1. 摄像头图像 → CLIP/SigLIP 视觉编码器 → 视觉特征
  2. 语言指令 → LLM Tokenizer → 语言特征
  3. 视觉特征 + 语言特征 → 拼接/交叉注意力 → Transformer
  4. → 输出动作

关键: VLA 用的是 CLIP 的视觉编码器（不是整个 CLIP）
     → CLIP 预训练的视觉编码器能理解语义（不仅是识别物体）
     → 这就是为什么 RT-2/OpenVLA 用 SigLIP 做视觉 backbone
```

**自检**: 为什么 VLA 用 CLIP 的视觉编码器而不是裸 ViT？好处是什么？

---

## 模块 4: SigLIP — VLA 的视觉标配（1 小时）

### 4.1 SigLIP vs CLIP — 20min

```
CLIP 的问题:
  需要全局 softmax 归一化 → 需要大 batch → 分布式训练

SigLIP 的改进:
  SigLIP = Sigmoid Loss for Language-Image Pre-training
  把 softmax 交叉熵 → 逐对 sigmoid 二分类

  CLIP:  L = -log(exp(s_ii) / Σ exp(s_ij))  ← 需要全局归一化
  SigLIP: L = -log(σ(s_ii)) - Σ log(σ(-s_ij))  ← 每对独立

  好处: 不依赖 batch size！更高效！
```

### 4.2 在 OpenVLA 中的使用 — 25min

```python
from transformers import AutoModel, AutoProcessor

# OpenVLA 实际使用的视觉编码器
vision_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")

# 查看架构
print(vision_model.config)
# {
#   "model_type": "siglip_vision_model",
#   "hidden_size": 1152,
#   "image_size": 384,
#   "patch_size": 14,
#   "num_hidden_layers": 27,
#   "num_attention_heads": 16,
# }

# 这本质上就是一个 ViT-L, 只是训练方式不同
```

### 4.3 视觉特征提取流程 — 15min

```python
# VLA 中视觉编码的完整流程
def encode_image_for_vla(image, vision_model, processor):
    """
    image: PIL Image 或 [H, W, C] numpy
    返回: 视觉特征，可直接送入 LLM
    """
    # 1. 预处理
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # [1, 3, 384, 384]

    # 2. 提取特征
    with torch.no_grad():
        vision_outputs = vision_model(pixel_values)  # [1, 729, 1152]
        # 729 = (384/14)^2 patches

    # 3. 可选：投影到 LLM 的维度
    projected = vision_projection(vision_outputs)  # [1, 729, llm_dim]

    return projected
```

---

## 模块 5: HuggingFace 视觉模型实战（1.5 小时）

### 5.1 加载各种视觉模型 — 30min

```python
from transformers import (
    AutoImageProcessor, AutoModel,
    CLIPProcessor, CLIPModel,
)
from PIL import Image

# ===== ViT =====
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224")

image = Image.open("robot_view.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
print(f"ViT output: {outputs.last_hidden_state.shape}")  # [1, 197, 768]

# ===== CLIP =====
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = clip_processor(
    images=image,
    text=["pick up the red block", "move left", "open drawer"],
    return_tensors="pt", padding=True,
)
outputs = clip_model(**inputs)
similarity = outputs.logits_per_image  # [1, 3] 图像与每个文本的相似度
print(f"Most similar text: {similarity.argmax()}")

# ===== SigLIP =====
siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
siglip_processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
```

### 5.2 特征可视化 — 30min

```python
import matplotlib.pyplot as plt

def visualize_attention(model, image, processor):
    """可视化 ViT 的注意力图"""
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)

    # 取最后一层的平均注意力
    attn = outputs.attentions[-1].mean(dim=1)  # [1, 197, 197]

    # CLS token 对各 patch 的注意力
    cls_attn = attn[0, 0, 1:].reshape(14, 14).detach().numpy()

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(cls_attn, cmap='hot')
    plt.title("CLS Attention Map")
    plt.show()

# 运行这个函数，观察模型到底在"看"什么
```

### 5.3 用视觉模型做简单特征匹配 — 30min

```python
# 实战: 构建一个简单的视觉-语言匹配器
# 给定一张机器人视角的图片和多个候选指令，找出最匹配的

class SimpleVisionLanguageMatcher:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def match(self, image, candidates: list[str]) -> str:
        inputs = self.processor(images=image, text=candidates,
                                return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        best_idx = outputs.logits_per_image.argmax().item()
        return candidates[best_idx]

# 测试
matcher = SimpleVisionLanguageMatcher()
# 假设你有一张"桌子上有红色积木"的图片
# result = matcher.match(image, ["pick up the red block", "move left", "stop"])
# print(f"模型认为指令是: {result}")
```

---

## 🎤 费曼挑战（1.5 小时）

### 任务 1: 画图讲解 ViT (30min)
在一张纸上画出 ViT 的完整数据流，每一步标注 shape:

```
[3, 224, 224] 原始图像
  → Patch Embed: Conv2d(3, 768, 16, 16) → ?
  → Flatten + Transpose → ?
  → Prepend CLS Token → ?
  → Add Position Embedding → ?
  → Transformer Blocks × 12 → ?
  → Extract CLS → ?
  → Linear Head → ?
```

### 任务 2: CLIP 对比学习讲解 (20min)
向一个不懂 ML 的朋友解释: CLIP 是怎么做到"看到狗就知道这是狗"的？

### 任务 3: VLA 视觉栈分析 (20min)
打开 OpenVLA 的 `prismatic/vla/vision_backbone/` 目录（clone 回来的代码），找到视觉编码器的配置，书面回答:

1. 他们用的是哪个视觉模型？
2. 图像分辨率是多少？为什么选这个分辨率？
3. Patch Size 是多少？输出多少个 patch？
4. 视觉特征是如何和语言特征拼接的？

### 任务 4: 面试题 (20min)
**"VLA 模型中为什么用 SigLIP 而不是普通的 ViT？两者有什么本质区别？"**

---

## 📝 今日自检清单

- [ ] 我能画出 ViT 的完整架构图（凭记忆）
- [ ] 我理解 Patch Embedding 的原理
- [ ] 我知道 CLS Token 的作用
- [ ] 我能解释 CLIP 的对比学习训练方式
- [ ] 我知道 SigLIP 相对于 CLIP 的改进
- [ ] 我成功加载并使用了 HuggingFace 视觉模型
- [ ] 我分析了 OpenVLA 的视觉编码器配置
- [ ] 我完成了面试题回答

---

> [!info] 知识库关联
> - [[../../../01-基础理论/视觉基础模型/ViT/ViT概述|ViT概述]] — 范式转变 + ViT架构 + VLA中的视觉
> - [[../../../01-基础理论/视觉基础模型/ViT/庖丁解牛VIT|庖丁解牛ViT]] — ViT代码级深度解析
> - [[../../../01-基础理论/视觉基础模型/CLIP/CLIP与多模态对齐|CLIP与多模态对齐]] — SigLIP vs CLIP + OpenVLA配置
> - [[../../../01-基础理论/视觉基础模型/DINO/DINOv2与自监督视觉|DINOv2与自监督视觉]] — 自监督 + 3D几何理解
>
> ✅ **完成打卡**: 视觉模块已掌握！明天进入具身智能核心领域！
>
> 🔜 **明日预告**: 具身智能概览 — 机器人基础、学习范式、关键数据集！
