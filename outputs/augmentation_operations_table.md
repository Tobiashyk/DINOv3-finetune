# DINOv3 DataAugmentationDINO - 图像处理操作表

## 每种裁剪类型的完整处理流程

### 表1：主要裁剪的处理步骤

| 裁剪类型 | 步骤1：几何变换 | 步骤2：颜色抖动 | 步骤3：模糊/反色 | 步骤4：归一化 | 输出形状 | 数量 |
|---------|----------------|----------------|-----------------|--------------|---------|------|
| **全局裁剪1** | RandomResizedCrop(224, scale=(0.32, 1.0))<br>+ RandomHorizontalFlip(p=0.5) | ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1) @ 80%<br>+ RandomGrayscale(p=0.2)<br>*(如果不共享)* | **GaussianBlur(p=1.0)**<br>*总是应用！* | ToImage()<br>+ ToDtype(float32)<br>+ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) | [3, 224, 224] | 1 |
| **全局裁剪2** | RandomResizedCrop(224, scale=(0.32, 1.0))<br>+ RandomHorizontalFlip(p=0.5) | ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1) @ 80%<br>+ RandomGrayscale(p=0.2)<br>*(如果不共享)* | GaussianBlur(p=0.1)<br>+ **RandomSolarize(threshold=128, p=0.2)** | ToImage()<br>+ ToDtype(float32)<br>+ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) | [3, 224, 224] | 1 |
| **局部裁剪** | RandomResizedCrop(96, scale=(0.05, 0.32))<br>+ RandomHorizontalFlip(p=0.5) | ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1) @ 80%<br>+ RandomGrayscale(p=0.2)<br>*(如果不共享)* | GaussianBlur(p=0.5) | ToImage()<br>+ ToDtype(float32)<br>+ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) | [3, 96, 96] | 8 |

---

### 表2：可选裁剪的处理步骤

| 裁剪类型 | 触发条件 | 处理步骤 | 输出形状 | 数量 |
|---------|---------|---------|---------|------|
| **Teacher全局裁剪** | `teacher_no_color_jitter=True` | 1. 使用 `im1_base` 和 `im2_base`（仅几何变换后）<br>2. 跳过颜色抖动<br>3. 跳过模糊/反色<br>4. 仅应用归一化 | [3, 224, 224] | 2 |
| **Teacher全局裁剪** | `teacher_no_color_jitter=False`（默认） | 与全局裁剪1和2相同（复用相同的张量） | [3, 224, 224] | 2 |
| **Gram Teacher裁剪** | `gram_teacher_crops_size != None`<br>+ `gram_teacher_no_distortions=True` | 1. 使用 `im1_base` 和 `im2_base`<br>2. 调整大小到 `gram_teacher_crops_size`<br>3. 仅应用归一化 | [3, gram_size, gram_size] | 2 |
| **Gram Teacher裁剪** | `gram_teacher_crops_size != None`<br>+ `gram_teacher_no_distortions=False` | 1. 使用 `global_crop_1_transf` 和 `global_crop_2_transf`（所有扭曲后）<br>2. 调整大小到 `gram_teacher_crops_size` | [3, gram_size, gram_size] | 2 |
| **局部裁剪（子集模式）** | `local_crops_subset_of_global_crops=True` | 1. 对 `im1_base` 和 `im2_base` 应用 local_transfo（各4个裁剪）<br>2. 随机提取 `local_crops_size × local_crops_size` 的补丁<br>3. 提取位置对齐到 `patch_size` 边界<br>4. 记录每个裁剪的偏移量 (rx, ry) | [3, 96, 96] | 8 |

---

### 表3：增强概率汇总

| 操作 | 全局裁剪1 | 全局裁剪2 | 局部裁剪 | 目的 |
|-----|----------|----------|---------|------|
| **随机缩放裁剪** | 100%（范围：32-100%） | 100%（范围：32-100%） | 100%（范围：5-32%） | 空间增强 |
| **随机水平翻转** | 50% | 50% | 50% | 空间增强 |
| **颜色抖动** | 80% | 80% | 80% | 颜色增强 |
| **随机灰度化** | 20% | 20% | 20% | 颜色增强 |
| **高斯模糊** | **100%** ⭐ | 10% | 50% | 防止捷径学习 |
| **随机反色** | 0% | **20%** ⭐ | 0% | 额外扭曲 |

---

### 表4：详细操作参数

| 操作 | 参数 | 说明 |
|-----|------|------|
| **随机缩放裁剪（全局）** | `size=224`<br>`scale=(0.32, 1.0)`<br>`interpolation=BICUBIC` | 裁剪原图的32%-100%，调整大小到224×224 |
| **随机缩放裁剪（局部）** | `size=96`<br>`scale=(0.05, 0.32)`<br>`interpolation=BICUBIC` | 裁剪原图的5%-32%，调整大小到96×96 |
| **随机水平翻转** | `p=0.5` | 50%概率水平翻转 |
| **颜色抖动** | `brightness=0.4`<br>`contrast=0.4`<br>`saturation=0.2`<br>`hue=0.1` | 随机改变亮度（±40%）、对比度（±40%）、饱和度（±20%）、色调（±10%） |
| **随机灰度化** | `p=0.2` | 20%概率转换为灰度图 |
| **高斯模糊** | `kernel_size=23`<br>`sigma=(0.1, 2.0)` | 应用随机sigma的高斯模糊 |
| **随机反色** | `threshold=128`<br>`p=0.2` | 20%概率反转阈值以上的像素 |
| **归一化** | `mean=[0.485, 0.456, 0.406]`<br>`std=[0.229, 0.224, 0.225]` | ImageNet标准化 |

---

### 表5：处理流程对比

| 阶段 | 全局裁剪1 | 全局裁剪2 | 局部裁剪（×8） |
|-----|----------|----------|---------------|
| **输入** | 原始PIL图像 | 原始PIL图像 | 原始PIL图像 |
| **几何增强** | ✅ RandomResizedCrop(224, 0.32-1.0) + 翻转 | ✅ RandomResizedCrop(224, 0.32-1.0) + 翻转 | ✅ RandomResizedCrop(96, 0.05-0.32) + 翻转 |
| **结果** | `im1_base` | `im2_base` | 8个不同的裁剪 |
| **颜色抖动** | ✅ 80% ColorJitter + 20% 灰度化 | ✅ 80% ColorJitter + 20% 灰度化 | ✅ 80% ColorJitter + 20% 灰度化 |
| **模糊** | ✅ **100% 高斯模糊**（强） | ✅ 10% 高斯模糊（弱） | ✅ 50% 高斯模糊（中） |
| **反色** | ❌ 无 | ✅ **20% 反色** | ❌ 无 |
| **归一化** | ✅ ImageNet mean/std | ✅ ImageNet mean/std | ✅ ImageNet mean/std |
| **输出** | `global_crop_1` [3, 224, 224] | `global_crop_2` [3, 224, 224] | `local_crops[0-7]` [3, 96, 96] |
| **增强强度** | 🔴 **强**（总是模糊） | 🟡 **弱**（很少模糊，有时反色） | 🟠 **中等**（一半模糊） |

---

### 表6：输出字典结构

```python
output = {
    # 学生网络的主要裁剪
    "global_crops": [
        global_crop_1,  # [3, 224, 224] - 强增强（100%模糊）
        global_crop_2,  # [3, 224, 224] - 弱增强（10%模糊，20%反色）
    ],

    # 教师网络的裁剪（可选的干净版本）
    "global_crops_teacher": [
        teacher_crop_1,  # [3, 224, 224] - 可选：无颜色抖动
        teacher_crop_2,  # [3, 224, 224] - 可选：无颜色抖动
    ],

    # 学生网络的局部裁剪
    "local_crops": [
        local_crop_1,  # [3, 96, 96] - 中等增强（50%模糊）
        local_crop_2,  # [3, 96, 96]
        ...
        local_crop_8,  # [3, 96, 96]
    ],

    # 可选：用于纹理/风格学习的Gram teacher裁剪
    "gram_teacher_crops": [
        gram_crop_1,  # [3, gram_size, gram_size] - 可选：无扭曲
        gram_crop_2,  # [3, gram_size, gram_size]
    ],

    # 可选：用于iBOT掩码图像建模的偏移量
    "offsets": [
        (rx1, ry1),  # local_crop_1的偏移量
        (rx2, ry2),  # local_crop_2的偏移量
        ...
        (rx8, ry8),  # local_crop_8的偏移量
    ],

    # 遗留标志
    "weak_flag": True,
}
```

---

### 表7：设计原理

| 设计选择 | 原理 |
|---------|------|
| **非对称增强** | 不同的增强强度防止模型学习捷径 |
| **全局裁剪1：100%模糊** | 强增强迫使模型学习即使在严重扭曲下也能工作的鲁棒特征 |
| **全局裁剪2：弱增强** | 提供更清晰的图像视图，帮助模型学习细粒度细节 |
| **局部裁剪：中等增强** | 在鲁棒性和细节之间取得平衡，专注于局部特征 |
| **多裁剪策略** | 2个全局 + 8个局部 = 同一图像的10个不同视图，显著增加训练多样性 |
| **Teacher无颜色抖动** | 教师网络获得更干净的图像，为学生提供更稳定的目标 |
| **Gram Teacher无扭曲** | 保留纹理/风格信息用于Gram矩阵损失 |
| **局部裁剪子集模式** | 确保全局和局部裁剪之间的空间对齐，用于iBOT掩码预测 |

---

### 表8：典型配置示例

| 使用场景 | 配置 |
|---------|------|
| **标准DINOv3训练** | `global_crops_scale=(0.32, 1.0)`<br>`local_crops_scale=(0.05, 0.32)`<br>`local_crops_number=8`<br>`share_color_jitter=False`<br>`teacher_no_color_jitter=False` |
| **使用干净Teacher** | 同上 + `teacher_no_color_jitter=True` |
| **使用Gram损失** | 同上 + `gram_teacher_crops_size=224`<br>`gram_teacher_no_distortions=True` |
| **使用iBOT损失** | 同上 + `local_crops_subset_of_global_crops=True`<br>`patch_size=16` |
| **共享颜色抖动** | 同上 + `share_color_jitter=True`<br>（在分支前对原始图像应用一次颜色抖动） |

---

## 总结

**每个输入生成的图像总数：**
- **最少（默认）：** 10张图像（2个全局 + 8个局部）
- **使用teacher_no_color_jitter：** 12张图像（2个全局 + 2个teacher全局 + 8个局部）
- **使用gram_teacher_crops：** 14张图像（2个全局 + 2个teacher全局 + 2个gram + 8个局部）
- **最多（所有选项）：** 14张图像 + 偏移量

**关键洞察：** 非对称增强策略是DINOv3成功的关键。通过对不同裁剪应用不同的增强强度，模型学会提取在各种扭曲下都能工作的鲁棒特征，同时保持捕获细粒度细节的能力。

---

## 代码示例

### 基本使用

```python
from dinov3.data.augmentations import DataAugmentationDINO
from PIL import Image

# 创建增强器
augmentation = DataAugmentationDINO(
    global_crops_scale=(0.32, 1.0),      # 全局裁剪范围
    local_crops_scale=(0.05, 0.32),      # 局部裁剪范围
    local_crops_number=8,                 # 8个局部裁剪
    global_crops_size=224,                # 全局裁剪尺寸
    local_crops_size=96,                  # 局部裁剪尺寸
)

# 加载图像
image = Image.open('image.jpg').convert('RGB')

# 应用增强
output = augmentation(image)

# 获取不同的裁剪
global_crops = output["global_crops"]        # List[Tensor], len=2, shape=[3,224,224]
local_crops = output["local_crops"]          # List[Tensor], len=8, shape=[3,96,96]
global_crops_teacher = output["global_crops_teacher"]  # List[Tensor], len=2
```

### 高级配置

```python
# 使用干净的Teacher裁剪
augmentation = DataAugmentationDINO(
    global_crops_scale=(0.32, 1.0),
    local_crops_scale=(0.05, 0.32),
    local_crops_number=8,
    teacher_no_color_jitter=True,  # Teacher使用无颜色抖动的图像
)

# 使用Gram Teacher裁剪
augmentation = DataAugmentationDINO(
    global_crops_scale=(0.32, 1.0),
    local_crops_scale=(0.05, 0.32),
    local_crops_number=8,
    gram_teacher_crops_size=224,           # 启用Gram Teacher裁剪
    gram_teacher_no_distortions=True,      # Gram Teacher无扭曲
)

# 使用局部裁剪子集模式（用于iBOT）
augmentation = DataAugmentationDINO(
    global_crops_scale=(0.32, 1.0),
    local_crops_scale=(0.05, 0.32),
    local_crops_number=8,
    local_crops_subset_of_global_crops=True,  # 从全局裁剪中提取局部裁剪
    patch_size=16,                             # ViT patch大小
)

output = augmentation(image)
offsets = output["offsets"]  # 获取局部裁剪的偏移量
```

---

## 可视化示例

### 增强效果对比

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_augmentation(image_path):
    """可视化不同增强策略的效果"""
    image = Image.open(image_path).convert('RGB')

    augmentation = DataAugmentationDINO(
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
        local_crops_number=8,
    )

    output = augmentation(image)

    # 反归一化函数
    def denormalize(tensor):
        img = tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        return np.clip(img, 0, 1)

    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # 显示2个全局裁剪
    axes[0, 0].imshow(denormalize(output['global_crops'][0]))
    axes[0, 0].set_title('Global Crop 1\n(100% Blur)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denormalize(output['global_crops'][1]))
    axes[0, 1].set_title('Global Crop 2\n(10% Blur, 20% Solarize)')
    axes[0, 1].axis('off')

    # 显示前8个局部裁剪
    for i in range(8):
        row = i // 5
        col = i % 5 + (2 if row == 0 else 0)
        axes[row, col].imshow(denormalize(output['local_crops'][i]))
        axes[row, col].set_title(f'Local Crop {i+1}\n(50% Blur)')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_visualization.png', dpi=150)
    plt.show()

# 使用示例
visualize_augmentation('data/koala_test/koala_0.png')
```

---

## 性能考虑

### 内存使用

| 配置 | 每张图像的内存占用（估算） |
|------|--------------------------|
| 默认（2全局 + 8局部） | ~50 MB |
| + Teacher裁剪 | ~75 MB |
| + Gram Teacher裁剪 | ~100 MB |
| 完整配置 | ~125 MB |

### 计算时间

| 操作 | 相对耗时 |
|------|---------|
| 几何变换 | 中等 |
| 颜色抖动 | 低 |
| 高斯模糊 | 高 |
| 归一化 | 低 |

**优化建议：**
- 使用 `share_color_jitter=True` 可以减少重复的颜色抖动计算
- 考虑使用更小的 `local_crops_number`（如4或6）来减少内存和计算
- 在调试时可以临时禁用高斯模糊以加快速度

---

## 常见问题

### Q1: 为什么全局裁剪1总是应用模糊？
**A:** 这是DINO的核心设计。强增强防止模型学习简单的捷径（如颜色直方图匹配），迫使它学习更深层的语义特征。

### Q2: 局部裁剪的作用是什么？
**A:** 局部裁剪帮助模型学习细粒度的局部特征，而不仅仅是全局语义。这对于密集预测任务（如分割、检测）特别重要。

### Q3: 什么时候使用 `local_crops_subset_of_global_crops=True`？
**A:** 当你使用iBOT损失（掩码图像建模）时。这确保局部裁剪与全局裁剪在空间上对齐，使得掩码预测任务有意义。

### Q4: Teacher裁剪和Student裁剪有什么区别？
**A:** 默认情况下它们相同。但设置 `teacher_no_color_jitter=True` 后，Teacher获得更干净的图像，提供更稳定的学习目标。

### Q5: 如何调整增强强度？
**A:** 可以修改：
- `global_crops_scale` 和 `local_crops_scale` 调整裁剪范围
- ColorJitter的参数调整颜色扭曲程度
- GaussianBlur的概率调整模糊频率

---

## 参考文献

1. **DINOv2 论文**: "DINOv2: Learning Robust Visual Features without Supervision" (2023)
2. **DINO 论文**: "Emerging Properties in Self-Supervised Vision Transformers" (2021)
3. **iBOT 论文**: "iBOT: Image BERT Pre-Training with Online Tokenizer" (2022)

---

**文档版本**: v1.0
**最后更新**: 2026-02-04
**作者**: Claude Code Assistant
