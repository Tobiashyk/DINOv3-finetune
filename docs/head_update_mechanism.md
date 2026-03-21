# DINOv3 Head 更新机制详解

## 🎯 快速回答

**是的，两个head（DINO head 和 iBOT head）都会更新！**

- **Student的head**: 通过梯度下降更新 ✅
- **Teacher的head**: 通过EMA（指数移动平均）更新 ✅

---

## 📋 详细机制

### 1️⃣ Student Head 的更新（梯度下降）

#### **参数包含**

从 `ssl_meta_arch.py:781-786` 可以看到：

```python
def get_params_groups(self):
    all_params_groups = []
    for name, m in self.student.items():  # 遍历所有组件
        logger.info(f"Getting parameter groups for {name}")
        all_params_groups += self.get_maybe_fused_params_for_submodel(m)
    return all_params_groups
```

`self.student.items()` 包含：
- `"backbone"`: Vision Transformer
- `"dino_head"`: DINO projection head
- `"ibot_head"`: iBOT prediction head

#### **参数配置**

从 `param_groups.py:56-108` 可以看到head的特殊配置：

```python
def get_params_groups_with_decay(model, lr_decay_rate=1.0,
                                  patch_embed_lr_mult=1.0,
                                  dino_head_wd_multiplier=1.0):
    all_param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 获取layer-wise learning rate decay
        decay_rate = get_vit_lr_decay_rate(
            name, lr_decay_rate, num_layers=n_blocks
        )

        d = {
            "name": name,
            "params": param,
            "lr_multiplier": decay_rate,  # 默认根据层数衰减
            "wd_multiplier": 1.0,         # 默认weight decay
        }

        # ✅ DINO Head 特殊设置
        if "dino_head" in name:
            d["wd_multiplier"] = dino_head_wd_multiplier  # 可配置

        # ✅ Last layer (归一化层)
        if "last_layer" in name:
            d["is_last_layer"] = True

        # ❌ 偏置和归一化参数不使用weight decay
        if name.endswith("bias") or "norm" in name or "gamma" in name:
            d["wd_multiplier"] = 0.0

        all_param_groups.append(d)
```

#### **学习率策略**

| 组件 | Learning Rate | Weight Decay | 说明 |
|-----|---------------|--------------|------|
| **Backbone (layer 0)** | base_lr × 0.65^24 | base_wd | Patch embedding最浅层 |
| **Backbone (layer 12)** | base_lr × 0.65^12 | base_wd | 中间层 |
| **Backbone (layer 24)** | base_lr × 0.65^0 = base_lr | base_wd | 最深层 |
| **DINO Head** | base_lr | base_wd × `dino_head_wd_multiplier` | **无衰减！** |
| **iBOT Head** | base_lr | base_wd | **无衰减！** |

**关键点**：
```python
# param_groups.py:77-83
decay_rate = get_vit_lr_decay_rate(name, lr_decay_rate, num_layers=n_blocks)

# get_vit_lr_decay_rate() 返回
layer_id = num_layers + 1  # 对于head，layer_id默认为最大值
return lr_decay_rate ** (num_layers + 1 - layer_id)
# 结果: 0.65^0 = 1.0 (无衰减!)
```

**Head使用最大的学习率**，因为它们不在backbone内部，不受layer-wise decay影响。

#### **更新过程**

```python
# 训练循环
for iteration, batch in enumerate(dataloader):
    optimizer.zero_grad()

    # 前向传播 (包含head)
    teacher_out = get_teacher_output(...)
    student_global, student_local = get_student_output(...)

    # 计算损失
    loss = compute_losses(teacher_out, student_global, student_local, ...)

    # 反向传播 (ssl_meta_arch.py:710-711)
    def backprop_loss(self, loss):
        loss.backward()  # ✅ 梯度传播到student的所有参数，包括head

    # 优化器更新
    optimizer.step()  # ✅ 更新student.dino_head和student.ibot_head
```

---

### 2️⃣ Teacher Head 的更新（EMA）

#### **EMA 更新机制**

从 `ssl_meta_arch.py:713-726`：

```python
def update_ema(self, m):
    """
    m: EMA momentum (通常是0.996)
    """
    if self.ema_params_lists is None:
        # 第一次调用时构建参数列表
        student_param_list = []
        teacher_param_list = []

        # ✅ 遍历所有组件（包括head！）
        for k in self.student.keys():  # ["backbone", "dino_head", "ibot_head"]
            for ms, mt in zip(
                self.student[k].parameters(),
                self.model_ema[k].parameters()  # model_ema = teacher
            ):
                student_param_list.append(ms)
                teacher_param_list.append(mt)

        self.ema_params_lists = (student_param_list, teacher_param_list)
    else:
        student_param_list, teacher_param_list = self.ema_params_lists

    # EMA 更新
    with torch.no_grad():
        # teacher = m * teacher + (1-m) * student
        torch._foreach_mul_(teacher_param_list, m)
        torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)
```

#### **EMA 动量调度**

```python
# 典型的动量调度 (从DINOv2)
def get_ema_momentum(iteration, max_iterations):
    base_momentum = 0.996
    final_momentum = 1.0

    # 从0.996线性增长到1.0
    m = base_momentum + (final_momentum - base_momentum) * iteration / max_iterations
    return m

# 例子:
# iteration=0:           m=0.996  (更新快，teacher变化快)
# iteration=max_iter/2:  m=0.998  (更新变慢)
# iteration=max_iter:    m=1.0    (几乎不更新，teacher趋于稳定)
```

#### **更新频率**

```python
# 每个训练batch后都更新
for iteration, batch in enumerate(dataloader):
    # ... 前向传播和优化器更新 ...

    # ✅ EMA更新 (每个batch都执行)
    m = get_ema_momentum(iteration)
    model.update_ema(m)

    # 包括:
    # - teacher.backbone.parameters()
    # - teacher.dino_head.parameters()  ✅
    # - teacher.ibot_head.parameters()  ✅
```

---

### 3️⃣ Head 初始化

从 `ssl_meta_arch.py:300-307`：

```python
def init_weights(self) -> None:
    # ✅ Student head 初始化
    self.student.backbone.init_weights()
    self.student.dino_head.init_weights()
    self.student.ibot_head.init_weights()

    # ✅ Teacher head 从 student 复制
    self.model_ema.load_state_dict(self.student.state_dict())
    # 这会复制:
    # - backbone weights
    # - dino_head weights ✅
    # - ibot_head weights ✅
```

---

## 🔍 代码验证

### 验证1: Student Head 在优化器中

```python
# 打印优化器中的参数
model = SSLMetaArch(cfg)
param_groups = model.get_params_groups()

for group in param_groups:
    print(f"{group['name']}: lr_mult={group['lr_multiplier']:.4f}, "
          f"wd_mult={group['wd_multiplier']:.4f}")

# 输出示例:
# backbone.patch_embed.proj.weight: lr_mult=0.0074, wd_mult=1.0
# backbone.blocks.23.norm1.weight: lr_mult=1.0000, wd_mult=0.0
# dino_head.mlp.0.weight: lr_mult=1.0000, wd_mult=0.1  ✅
# dino_head.last_layer.weight_v: lr_mult=1.0000, wd_mult=0.0  ✅
# ibot_head.mlp.0.weight: lr_mult=1.0000, wd_mult=1.0  ✅
```

### 验证2: Teacher Head 在 EMA 中

```python
# 在训练前检查
model = SSLMetaArch(cfg)
print("Student keys:", list(model.student.keys()))
# 输出: ['backbone', 'dino_head', 'ibot_head']

print("Teacher keys:", list(model.teacher.keys()))
# 输出: ['backbone', 'dino_head', 'ibot_head']  ✅

# EMA更新后检查
student_dino_weight_before = model.student.dino_head.mlp[0].weight.clone()
teacher_dino_weight_before = model.teacher.dino_head.mlp[0].weight.clone()

# 训练一个batch
optimizer.step()
model.update_ema(m=0.996)

student_dino_weight_after = model.student.dino_head.mlp[0].weight
teacher_dino_weight_after = model.teacher.dino_head.mlp[0].weight

# Teacher的变化应该是student变化的(1-0.996)=0.004倍
print("Student head weight changed:",
      (student_dino_weight_after - student_dino_weight_before).abs().max())
print("Teacher head weight changed:",
      (teacher_dino_weight_after - teacher_dino_weight_before).abs().max())
# 输出:
# Student head weight changed: 0.0123
# Teacher head weight changed: 0.0000  ✅ (EMA让变化变慢)
```

---

## 📊 Head 架构细节

### DINO Head 结构

从 `dinov3/layers/dino_head.py`：

```python
class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,           # 768 (ViT-L)
        out_dim,          # 65536 (prototypes)
        hidden_dim=2048,
        bottleneck_dim=256,
        nlayers=3,
        norm_last_layer=True,
    ):
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),      # 768 → 2048
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),  # 2048 → 2048
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),  # 2048 → 256
        )

        # Last layer (normalized weight)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)  # 256 → 65536
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False  # ❌ 固定norm
```

**参数统计**：
- `mlp.0.weight`: [2048, 768] = 1,572,864
- `mlp.2.weight`: [2048, 2048] = 4,194,304
- `mlp.4.weight`: [256, 2048] = 524,288
- `last_layer.weight_v`: [256, 65536] = 16,777,216
- **总计**: ~23M 参数

### iBOT Head 结构

```python
# 使用相同的DINOHead架构，但不同的输出维度
ibot_head = DINOHead(
    in_dim=768,
    out_dim=8192,  # 更少的prototypes
    hidden_dim=2048,
    bottleneck_dim=256,
    nlayers=3,
)
```

**参数统计**：
- 前面层相同
- `last_layer.weight_v`: [256, 8192] = 2,097,152
- **总计**: ~8M 参数

---

## ⚙️ 配置选项

### Head 相关配置

```yaml
# config.yaml
dino:
  head_n_prototypes: 65536      # DINO head输出维度
  head_hidden_dim: 2048          # MLP隐藏层维度
  head_bottleneck_dim: 256       # Bottleneck维度
  head_nlayers: 3                # MLP层数
  head_norm_last_layer: true     # 是否固定last layer的norm

ibot:
  head_n_prototypes: 8192        # iBOT head输出维度
  head_hidden_dim: 2048
  head_bottleneck_dim: 256
  head_nlayers: 3
  head_norm_last_layer: true

optim:
  base_lr: 0.0001                # 基础学习率
  weight_decay: 0.04             # 基础weight decay
  dino_head_wd_multiplier: 0.1   # ✅ DINO head的weight decay倍数
  layerwise_decay: 0.65          # Backbone的layer-wise衰减率
```

### 实际使用的Weight Decay

```python
# Backbone参数
backbone_wd = base_wd * 1.0 = 0.04

# DINO Head参数
dino_head_wd = base_wd * dino_head_wd_multiplier = 0.04 * 0.1 = 0.004

# iBOT Head参数
ibot_head_wd = base_wd * 1.0 = 0.04

# Bias和Norm参数
bias_norm_wd = 0.0  # 不使用weight decay
```

**为什么DINO Head的weight decay更小？**
- DINO head直接输出分类logits，过强的正则化会损害性能
- iBOT head输出用于重建，可以使用更强的正则化

---

## 🔄 完整更新流程图

```
训练开始
  │
  ├─ 初始化 Student (随机或预训练)
  │  ├─ student.backbone
  │  ├─ student.dino_head  ✅ 随机初始化
  │  └─ student.ibot_head  ✅ 随机初始化
  │
  ├─ 初始化 Teacher (从Student复制)
  │  ├─ teacher.backbone ← student.backbone
  │  ├─ teacher.dino_head ← student.dino_head  ✅ 复制
  │  └─ teacher.ibot_head ← student.ibot_head  ✅ 复制
  │
  └─ 训练循环:
     │
     ├─ 前向传播
     │  ├─ Teacher: backbone → dino_head, ibot_head
     │  └─ Student: backbone → dino_head, ibot_head
     │
     ├─ 计算损失
     │  ├─ DINO Loss: student.dino_head vs teacher.dino_head
     │  └─ iBOT Loss: student.ibot_head vs teacher.ibot_head
     │
     ├─ 反向传播
     │  └─ loss.backward()
     │     └─ 梯度传播到: student.backbone, student.dino_head, student.ibot_head  ✅
     │
     ├─ 优化器更新
     │  └─ optimizer.step()
     │     ├─ student.backbone.parameters() -= lr * grad  ✅
     │     ├─ student.dino_head.parameters() -= lr * grad  ✅
     │     └─ student.ibot_head.parameters() -= lr * grad  ✅
     │
     └─ EMA 更新
        └─ update_ema(m=0.996)
           ├─ teacher.backbone = 0.996*teacher.backbone + 0.004*student.backbone  ✅
           ├─ teacher.dino_head = 0.996*teacher.dino_head + 0.004*student.dino_head  ✅
           └─ teacher.ibot_head = 0.996*teacher.ibot_head + 0.004*student.ibot_head  ✅
```

---

## 💡 常见问题

### Q1: 为什么Teacher的head也需要更新？

**A**: Teacher的head输出soft targets，如果不更新：
- Head参数会与backbone不匹配
- 输出的prototypes分布会失效
- 训练会不稳定甚至崩溃

### Q2: 能否只更新Student的head？

**A**: 不行！Teacher必须通过EMA保持与Student的一致性：
```python
# ❌ 错误做法
teacher.backbone ← EMA(student.backbone)
teacher.dino_head ← [frozen]  # 不更新

# ✅ 正确做法
teacher.backbone ← EMA(student.backbone)
teacher.dino_head ← EMA(student.dino_head)
```

### Q3: Head的学习率为什么不衰减？

**A**: Head不是ViT的一部分，需要快速适应：
- Backbone学习缓慢（layer-wise decay）
- Head学习快速（full learning rate）
- 平衡两者的学习速度

### Q4: 为什么DINO Head的weight decay更小？

**A**:
```python
# DINO head直接影响对比学习
dino_head_wd = 0.004  # 较小，允许更大的权重

# iBOT head用于重建任务
ibot_head_wd = 0.04   # 较大，防止过拟合
```

### Q5: Last layer的weight norm是什么？

**A**: Weight Normalization将权重分解为：
```python
w = g * (v / ||v||)  # g是标量，v是向量

# 在DINOHead中:
self.last_layer = weight_norm(Linear(...))
self.last_layer.weight_g.requires_grad = False  # ❌ 固定g=1

# 只学习方向v，不学习大小g
# 好处: 输出logits的scale保持稳定
```

---

## 📈 监控 Head 更新

### TensorBoard 日志

```python
# 在训练循环中
if iteration % log_freq == 0:
    # Student head norm
    student_dino_norm = sum(
        p.norm() for p in model.student.dino_head.parameters()
    )
    logger.log("student/dino_head_norm", student_dino_norm)

    # Teacher head norm
    teacher_dino_norm = sum(
        p.norm() for p in model.teacher.dino_head.parameters()
    )
    logger.log("teacher/dino_head_norm", teacher_dino_norm)

    # Head梯度
    student_dino_grad = sum(
        p.grad.norm() for p in model.student.dino_head.parameters()
        if p.grad is not None
    )
    logger.log("student/dino_head_grad_norm", student_dino_grad)
```

### 预期行为

```
训练初期 (iteration < 1000):
- student_dino_head_grad_norm: 0.5 ~ 2.0  (较大)
- teacher_dino_head_norm: 快速增长
- dino_loss: 快速下降

训练中期 (1000 < iteration < 10000):
- student_dino_head_grad_norm: 0.1 ~ 0.5  (减小)
- teacher_dino_head_norm: 稳定增长
- dino_loss: 缓慢下降

训练后期 (iteration > 10000):
- student_dino_head_grad_norm: 0.01 ~ 0.1  (很小)
- teacher_dino_head_norm: 趋于稳定
- dino_loss: 接近收敛
```

---

## ✅ 总结

### Head 更新机制

| Head | 更新方式 | 更新频率 | 学习率 | Weight Decay | 梯度 |
|------|---------|---------|--------|-------------|-----|
| **Student DINO** | 梯度下降 | 每个batch | base_lr × 1.0 | base_wd × 0.1 | ✅ |
| **Student iBOT** | 梯度下降 | 每个batch | base_lr × 1.0 | base_wd × 1.0 | ✅ |
| **Teacher DINO** | EMA | 每个batch | - | - | ❌ |
| **Teacher iBOT** | EMA | 每个batch | - | - | ❌ |

### 关键代码位置

- **参数组设置**: `param_groups.py:56-108`
- **学习率衰减**: `param_groups.py:12-53`
- **EMA更新**: `ssl_meta_arch.py:713-726`
- **梯度反向传播**: `ssl_meta_arch.py:710-711`
- **优化器配置**: `ssl_meta_arch.py:781-786`

### 设计原则

1. **Head使用全学习率**: 不受layer-wise decay影响
2. **DINO Head轻度正则**: weight decay × 0.1
3. **iBOT Head正常正则**: weight decay × 1.0
4. **Teacher通过EMA同步**: 保持与Student的一致性
5. **Last layer固定norm**: 稳定输出scale

**结论**: 两个head都会积极更新，是模型训练的关键组件！
