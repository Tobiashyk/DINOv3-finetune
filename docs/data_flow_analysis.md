# DINOv3 数据处理流程详解

## 📊 完整数据流程图

```
原始图像 (ImageNet)
    │
    ├─── DataAugmentationDINO.__call__()
    │
    ├──────────────────┬────────────────┬─────────────────┐
    │                  │                │                 │
    v                  v                v                 v
全局Crop 1         全局Crop 2      局部Crops       Gram Teacher Crops
(224×224)         (224×224)      (N × 96×96)         (可选, 518×518)
    │                  │                │                 │
    └──────┬───────────┘                │                 │
           │                            │                 │
           v                            v                 v
    [2, B, 3, 224, 224]        [N*B, 3, 96, 96]   [2, B, 3, 518, 518]
           │                            │                 │
           │                            │                 │
    ┌──────┴────────┐                  │                 │
    │               │                  │                 │
    v               v                  v                 v
Teacher 前向    Student 前向       Student 前向    Gram Teacher 前向
(无梯度)        (全局crops)        (局部crops)       (无梯度)
    │               │                  │                 │
    └───────────────┴──────────────────┴─────────────────┘
                    │
                    v
            计算多任务损失 (Loss Computation)
                    │
        ┌───────────┼───────────┬─────────────┐
        │           │           │             │
        v           v           v             v
    DINO Loss   iBOT Loss   KoLeo Loss   Gram Loss
     (对比)       (掩码)      (正则)       (特征)
        │           │           │             │
        └───────────┴───────────┴─────────────┘
                    │
                    v
            Total Loss → Backward
```

---

## 1️⃣ 数据增强阶段 (DataAugmentationDINO)

### 输入
- **原始图像**: 1张 RGB 图像 (任意分辨率)

### 输出
```python
{
    "global_crops": [crop1, crop2],              # 2个全局crops
    "global_crops_teacher": [crop1_t, crop2_t],  # Teacher用的全局crops（可选去色彩抖动）
    "local_crops": [crop1, ..., cropN],          # N个局部crops (默认10个)
    "gram_teacher_crops": [gram1, gram2],        # Gram teacher crops (可选)
    "offsets": [(x1, y1), ..., (xN, yN)],        # 局部crops的位置偏移
}
```

### 增强策略

#### 全局 Crops (224×224)
```python
# 步骤1: 随机裁剪和翻转
im1_base = RandomResizedCrop(224, scale=(0.32, 1.0))(image)
im2_base = RandomResizedCrop(224, scale=(0.32, 1.0))(image)

# 步骤2: 颜色增强 (共享或独立)
if share_color_jitter:
    image = ColorJitter()(image)  # 全局共享

# 步骤3: 高斯模糊 + Solarization
crop1 = GaussianBlur(p=1.0)(im1_base)  # 第1个crop总是模糊
crop2 = Solarization(p=0.2)(im2_base)  # 第2个crop可能反色

# 步骤4: 标准化
crop1 = Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])(crop1)
```

#### 局部 Crops (96×96)
```python
# 两种模式:
# 模式1: 独立随机裁剪 (默认)
local_crops = [
    RandomResizedCrop(96, scale=(0.05, 0.32))(image)
    for _ in range(10)
]

# 模式2: 从全局crops中裁剪 (local_crops_subset_of_global_crops=True)
# 好处: 保证局部crops一定在全局crops内部
for i in range(10):
    rx = random.randint(0, (224-96) // 16) * 16
    ry = random.randint(0, (224-96) // 16) * 16
    local_crops[i] = global_crop[:, rx:rx+96, ry:ry+96]
    offsets[i] = (rx, ry)
```

#### Gram Teacher Crops (可选, 518×518)
```python
if gram_teacher_crops_size is not None:
    if gram_teacher_no_distortions:
        # 无颜色扭曲，直接resize
        gram_crop = Normalize(Resize(518)(im1_base))
    else:
        # 共享颜色扭曲，在增强后resize
        gram_crop = Resize(518)(global_crop_1)
```

---

## 2️⃣ Teacher 模型数据处理

### 代码位置
`ssl_meta_arch.py:432-474` → `get_teacher_output()`

### 输入
```python
images: [2, B, 3, 224, 224]  # 2个全局crops
mask_indices_list: [n_masked_patches]  # 学生模型中被掩码的patch索引
teacher_temp: float  # Teacher temperature (0.04 → 0.07 warmup)
```

### 处理流程

```python
# 1. Flatten crops维度
images = images.flatten(0, 1)  # [2*B, 3, 224, 224]

# 2. Backbone 前向传播 (无梯度)
with torch.no_grad():
    backbone_out = self.teacher.backbone(images, is_training=True)

    cls_tokens = backbone_out["x_norm_clstoken"]     # [2*B, 768]
    patch_tokens = backbone_out["x_norm_patchtokens"] # [2*B, 256, 768]
    # 256 = (224/14)^2 = 16^2 (ViT-L/14的patch数量)

# 3. DINO Head: 处理所有CLS tokens
cls_after_head = self.teacher.dino_head(cls_tokens)  # [2*B, 65536]
# 65536 是原型数量 (prototypes)

# 4. Sinkhorn-Knopp Centering (关键操作!)
cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
    cls_after_head,
    teacher_temp=0.04  # 初始温度
)  # [2*B, 65536] → 概率分布

# 5. iBOT Head: 只处理被掩码的patch tokens
# 关键: Teacher看完整图像，但只输出学生被掩码位置的预测
masked_patches = torch.index_select(
    patch_tokens.flatten(0, 1),  # [2*B*256, 768]
    dim=0,
    index=mask_indices_list  # 例如: [3520] (2*B*256的40%被掩码)
)
masked_patch_after_head = self.teacher.ibot_head(masked_patches)  # [3520, 8192]

# 6. Sinkhorn-Knopp Centering (patch-level)
masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
    masked_patch_after_head,
    teacher_temp=0.04,
    n_masked_patches_tensor=n_masked_patches  # 用于distributed normalization
)  # [3520, 8192]
```

### 输出
```python
{
    "cls_pre_head": [2, B, 768],           # Backbone输出的CLS token (用于可视化/分析)
    "reg_pre_head": [2, B, R, 768],        # Register tokens (DINOv3特有)
    "patch_pre_head": [2, B, 256, 768],    # Backbone输出的Patch tokens
    "cls_after_head": [2, B, 65536],       # DINO head输出 (未centered)
    "cls_centered": [2, B, 65536],         # ✅ DINO Loss的目标
    "masked_patch_centered": [3520, 8192], # ✅ iBOT Loss的目标
}
```

### 🔑 关键点

1. **Teacher看完整图像**: 无掩码，输入是干净的全局crops
2. **无梯度计算**: `@torch.no_grad()` 装饰器
3. **Sinkhorn-Knopp**:
   - 作用: 防止模型坍塌 (mode collapse)
   - 原理: 将logits转为均匀分布的soft assignment
   - 跨GPU同步: 使用distributed all-reduce
4. **只输出被掩码的patch**: 节省内存和计算

---

## 3️⃣ Student 模型数据处理

### 代码位置
`ssl_meta_arch.py:530-582` → `get_student_output()`

### 输入
```python
global_crops: [2, B, 3, 224, 224]  # 全局crops
local_crops: [N, B, 3, 96, 96]     # 局部crops (N=10)
masks: [2*B, 256]                   # 布尔掩码，True表示被掩码
mask_indices_list: [3520]           # 被掩码的patch索引 (扁平化)
```

### 处理流程

```python
# 1. Flatten crops维度
global_crops = global_crops.flatten(0, 1)  # [2*B, 3, 224, 224]
local_crops = local_crops.flatten(0, 1)    # [N*B, 3, 96, 96]

# 2. 联合前向传播 (关键优化!)
# 一次前向同时处理全局和局部crops，提高效率
global_out, local_out = self.student.backbone(
    [global_crops, local_crops],  # 列表输入
    masks=[masks, None],           # 只对全局crops应用掩码
    is_training=True
)

# 输出:
g_cls = global_out["x_norm_clstoken"]      # [2*B, 768]
g_patch = global_out["x_norm_patchtokens"] # [2*B, 256, 768]
l_cls = local_out["x_norm_clstoken"]       # [N*B, 768]
l_patch = local_out["x_norm_patchtokens"]  # [N*B, 36, 768]
# 36 = (96/14)^2 ≈ 6^2 (局部crop的patch数量)

# 3. iBOT Head: 只处理被掩码的patch
masked_patches = torch.index_select(
    g_patch.flatten(0, 1),     # [2*B*256, 768]
    dim=0,
    index=mask_indices_list    # [3520]
)
masked_patch_after_head = self.student.ibot_head(masked_patches)  # [3520, 8192]

# 4. DINO Head: 处理全局+局部的所有CLS tokens
buffer = torch.cat([
    g_cls,  # [2*B, 768]
    l_cls   # [N*B, 768]
], dim=0)  # [(2+N)*B, 768]

buffer = self.student.dino_head(buffer)  # [(2+N)*B, 65536]

# 分离全局和局部
global_cls_after_head = buffer[:2*B]   # [2*B, 65536]
local_cls_after_head = buffer[2*B:]    # [N*B, 65536]
```

### 输出
```python
global_out = {
    "cls_pre_head": [2, B, 768],           # 全局crops的CLS token
    "patch_pre_head": [2, B, 256, 768],    # 全局crops的patch tokens
    "cls_after_head": [2, B, 65536],       # ✅ DINO Loss的学生预测
    "masked_patch_after_head": [3520, 8192], # ✅ iBOT Loss的学生预测
    "masked_patch_pre_head": [3520, 768],  # 被掩码patch的特征 (用于分析)
}

local_out = {
    "cls_pre_head": [N, B, 768],           # 局部crops的CLS token
    "patch_pre_head": [N, B, 36, 768],     # 局部crops的patch tokens
    "cls_after_head": [N, B, 65536],       # ✅ DINO Loss的学生预测 (局部)
}
```

### 🔑 关键点

1. **联合前向传播**: 全局和局部crops在一次前向中处理，共享计算
2. **掩码应用**:
   - 只对全局crops的patch tokens应用掩码
   - 局部crops不使用掩码
3. **灵活的backbone输入**:
   ```python
   # DINOv3 backbone支持列表输入
   [input1, input2], masks=[mask1, mask2]
   ```
4. **Student有梯度**: 需要梯度进行反向传播

---

## 4️⃣ Gram Teacher 数据处理 (可选)

### 代码位置
`ssl_meta_arch.py:476-528` → `get_gram_teacher_output()`

### 两种模式

#### 模式1: 使用EMA Teacher (`gram.ema_teacher=True`)
```python
# 直接复用main teacher的输出，无需额外前向传播
teacher_patches = teacher_global["patch_pre_head"].flatten(0, 1)  # [2*B, 256, 768]
```

#### 模式2: 独立Gram Teacher (`gram.ema_teacher=False`)
```python
# 输入独立的gram crops (518×518)
images: [2, B, 3, 518, 518]

# 前向传播
with torch.no_grad():
    backbone_out = self.gram_teacher.backbone(images, is_training=True)
    teacher_patches = backbone_out["x_norm_patchtokens"]  # [2*B, 1369, 768]
    # 1369 = (518/14)^2 ≈ 37^2

# 下采样到student的分辨率
if teacher_patches.shape[1] != student_patches.shape[1]:
    N = 37  # 518/14
    N_student = 16  # 224/14

    # Reshape到空间维度
    patches_hw = teacher_patches.transpose(-2, -1).unflatten(-1, (N, N))
    # [2*B, 768, 37, 37]

    # 双线性插值下采样
    patches_hw = F.interpolate(
        patches_hw,
        size=(N_student, N_student),  # (16, 16)
        mode='bilinear',
        align_corners=False,
        antialias=True
    )

    # Reshape回序列格式
    teacher_patches = patches_hw.flatten(-2, -1).transpose(-2, -1)
    # [2*B, 256, 768]
```

### Token 选择策略

```python
# 配置: gram.tokens_used = "all" | "masked" | "unmasked"

if self.gram_tokens_used == "masked":
    # 只计算被掩码的tokens
    student_patches = student_patches[masks]  # [n_masked, 768]
    teacher_patches = teacher_patches[masks]  # [n_masked, 768]

elif self.gram_tokens_used == "unmasked":
    # 只计算未被掩码的tokens
    student_patches = student_patches[~masks]  # [n_unmasked, 768]
    teacher_patches = teacher_patches[~masks]  # [n_unmasked, 768]

else:  # "all"
    # 使用所有tokens
    pass  # [2*B, 256, 768]
```

### 输出
```python
{
    "student_patches": [2*B, 256, 768] or [n_selected, 768],  # 学生的patch特征
    "teacher_patches": [2*B, 256, 768] or [n_selected, 768],  # Gram teacher的patch特征
    "orig_student_patches": [2*B, 256, 768],                  # 原始特征（统计用）
    "orig_teacher_patches": [2*B, 256, 768],                  # 原始特征（统计用）
}
```

### 🔑 关键点

1. **高分辨率输入**: 518×518 vs 224×224
2. **独立更新策略**:
   - 可以从checkpoint加载
   - 定期从EMA teacher复制权重
3. **特征对齐**: 通过插值确保维度匹配

---

## 5️⃣ 数据尺寸总结表

| 数据类型 | Teacher | Student (全局) | Student (局部) | Gram Teacher |
|---------|---------|---------------|---------------|--------------|
| **输入图像** | [2*B, 3, 224, 224] | [2*B, 3, 224, 224] | [N*B, 3, 96, 96] | [2*B, 3, 518, 518] |
| **掩码** | ❌ 无 | ✅ [2*B, 256] | ❌ 无 | ❌ 无 |
| **CLS Token** | [2*B, 768] | [2*B, 768] | [N*B, 768] | ❌ 不使用 |
| **Patch Tokens** | [2*B, 256, 768] | [2*B, 256, 768] | [N*B, 36, 768] | [2*B, 256, 768] |
| **DINO Head输出** | [2*B, 65536] | [2*B, 65536] | [N*B, 65536] | ❌ 不使用 |
| **iBOT Head输出** | [n_masked, 8192] | [n_masked, 8192] | ❌ 不使用 | ❌ 不使用 |
| **梯度** | ❌ 无梯度 | ✅ 有梯度 | ✅ 有梯度 | ❌ 无梯度 |

其中:
- `B` = batch size per GPU (例如: 16)
- `N` = 局部crops数量 (默认: 10)
- `n_masked` = 被掩码的patch数量 (例如: 2*B*256*0.4 = 3276)
- `768` = ViT-L的embedding维度
- `256` = (224/14)^2, ViT-L/14的patch数量
- `36` = (96/14)^2 ≈ 6^2, 局部crop的patch数量
- `65536` = DINO head的原型数量
- `8192` = iBOT head的原型数量

---

## 6️⃣ 关键处理差异

### Teacher vs Student

| 特性 | Teacher | Student |
|-----|---------|---------|
| **输入crops** | 仅全局 (2个) | 全局 + 局部 (2+N个) |
| **掩码** | ❌ 看完整图像 | ✅ 全局crops被掩码 |
| **梯度** | ❌ 无梯度 (`@torch.no_grad()`) | ✅ 需要梯度 |
| **输出centering** | ✅ Sinkhorn-Knopp | ❌ 直接输出logits |
| **更新方式** | EMA更新 (`m=0.996`) | 梯度下降 |
| **作用** | 提供稳定的soft targets | 学习表示 |

### 全局 Crops vs 局部 Crops (Student)

| 特性 | 全局 Crops | 局部 Crops |
|-----|-----------|-----------|
| **尺寸** | 224×224 | 96×96 |
| **数量** | 2 | N (默认10) |
| **掩码** | ✅ 有 (40%) | ❌ 无 |
| **iBOT Loss** | ✅ 参与 | ❌ 不参与 |
| **DINO Loss** | ✅ 参与 | ✅ 参与 |
| **与Teacher比较** | ✅ 比较 | ❌ 仅内部对比 |

---

## 7️⃣ 损失计算中的数据使用

### DINO Loss (对比学习)
```python
# 全局 crops: student vs teacher
dino_global = cross_entropy(
    student_logits=student_global["cls_after_head"],  # [2, B, 65536]
    teacher_probs=teacher_global["cls_centered"],     # [2, B, 65536]
    ignore_diagonal=True  # 不与自己比较
)

# 局部 crops: student vs teacher
dino_local = cross_entropy(
    student_logits=student_local["cls_after_head"],   # [N, B, 65536]
    teacher_probs=teacher_global["cls_centered"],     # [2, B, 65536]
)

# 组合
dino_loss = dino_global * 0.5 + dino_local * 0.5
```

### iBOT Loss (掩码预测)
```python
# 只比较被掩码的patch tokens
ibot_loss = cross_entropy(
    student_logits=student_global["masked_patch_after_head"],  # [n_masked, 8192]
    teacher_probs=teacher_global["masked_patch_centered"],     # [n_masked, 8192]
)
```

### KoLeo Loss (特征正则化)
```python
# 使用student的CLS token (head之前)
koleo_loss = sum([
    koleo_loss_fn(student_global["cls_pre_head"][i])  # [B, 768]
    for i in range(2)  # 2个全局crops
]) / 2
```

### Gram Loss (特征一致性)
```python
# 使用patch-level特征
gram_loss = gram_loss_fn(
    student_patches=gram_global["student_patches"],  # [2*B, 256, 768]
    teacher_patches=gram_global["teacher_patches"],  # [2*B, 256, 768]
    img_level=True  # 图像级别计算，而非batch级别
)
```

---

## 8️⃣ 数据流动的时间线

```python
# 训练循环中的数据流
for iteration, batch in enumerate(dataloader):
    # ========== 步骤1: 数据准备 ==========
    data = {
        "collated_global_crops": [2*B, 3, 224, 224],
        "collated_local_crops": [N*B, 3, 96, 96],
        "collated_masks": [2*B, 256],  # 布尔掩码
        "mask_indices_list": [n_masked],  # 扁平化索引
        "collated_gram_teacher_crops": [2*B, 3, 518, 518],  # 可选
    }

    # ========== 步骤2: Teacher前向 (无梯度) ==========
    with torch.no_grad():
        teacher_out = get_teacher_output(
            data["collated_global_crops"],  # 只用全局crops
            mask_indices_list=data["mask_indices_list"],
        )

    # ========== 步骤3: Student前向 (有梯度) ==========
    student_global, student_local = get_student_output(
        global_crops=data["collated_global_crops"],
        local_crops=data["collated_local_crops"],
        masks=data["collated_masks"],
    )

    # ========== 步骤4: Gram Teacher前向 (无梯度, 可选) ==========
    if gram_use_loss:
        with torch.no_grad():
            gram_out = get_gram_teacher_output(
                data["collated_gram_teacher_crops"],
                teacher_global=teacher_out,
                student_global=student_global,
            )

    # ========== 步骤5: 计算损失 ==========
    loss = (
        dino_loss_weight * dino_loss(student_out, teacher_out) +
        ibot_loss_weight * ibot_loss(student_out, teacher_out) +
        koleo_loss_weight * koleo_loss(student_out) +
        gram_loss_weight * gram_loss(student_out, gram_out)
    )

    # ========== 步骤6: 反向传播 ==========
    loss.backward()  # 只更新student的梯度

    # ========== 步骤7: 优化器步进 ==========
    optimizer.step()

    # ========== 步骤8: 更新Teacher (EMA) ==========
    with torch.no_grad():
        for param_s, param_t in zip(student.parameters(), teacher.parameters()):
            param_t.data = m * param_t.data + (1 - m) * param_s.data

    # ========== 步骤9: 更新Gram Teacher (可选, 周期性) ==========
    if iteration % gram_update_frequency == 0:
        with torch.no_grad():
            gram_teacher.load_state_dict(teacher.state_dict())
```

---

## 9️⃣ 关键设计决策

### 为什么Teacher看完整图像？
- **避免信息泄漏**: 如果Teacher也被掩码，学生可能学会作弊（通过对齐掩码模式）
- **提供完整监督**: Teacher输出基于完整上下文的特征

### 为什么局部crops不用掩码？
- **已经足够困难**: 96×96的局部crops已经是图像的小部分
- **计算效率**: 减少不必要的计算

### 为什么需要Gram Teacher？
- **多尺度特征**: 提供更丰富的特征正则化
- **独立视角**: 与main teacher的EMA更新解耦

### 为什么联合前向传播？
```python
# 高效写法:
global_out, local_out = backbone([global_crops, local_crops])

# 低效写法 (慢2倍):
global_out = backbone(global_crops)
local_out = backbone(local_crops)
```
- **共享计算**: 某些操作（如LayerNorm）可以合并
- **减少kernel启动**: GPU并行效率更高

---

## 🎯 总结

### 数据处理的核心思想

1. **多视角学习**: 通过全局+局部crops提供不同尺度的视角
2. **Teacher-Student框架**: Teacher提供稳定的soft targets
3. **掩码策略**: 只对student的全局crops应用掩码
4. **联合计算**: 最大化计算效率
5. **灵活架构**: 支持可选的Gram teacher

### 关键数据流

```
原始图像
  ↓ (数据增强)
全局Crops (2×224²) + 局部Crops (10×96²) + Gram Crops (2×518²)
  ↓
Teacher (完整)    Student (掩码)    Gram Teacher (完整)
  ↓                    ↓                    ↓
CLS + Patch         CLS + Patch          Patch only
  ↓                    ↓                    ↓
DINO Head           DINO Head            -
iBOT Head           iBOT Head            -
  ↓                    ↓                    ↓
Sinkhorn-Knopp      Raw Logits           Raw Features
  ↓                    ↓                    ↓
      ↘               ↓                  ↙
           Multi-Task Loss
                ↓
           Backprop (仅Student)
                ↓
           EMA Update (Teacher)
```

### 性能考虑

- **内存**: ~40GB/GPU (ViT-L, batch=16)
- **计算**: ~350 images/sec (8×A100)
- **通信**: Sinkhorn-Knopp需要all-reduce
- **优化**: 混合精度 (BF16) + FSDP

希望这个详细的分析对你理解DINOv3的数据处理流程有帮助!
