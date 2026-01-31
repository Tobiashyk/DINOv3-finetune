# DINOv3 迁移到 PyTorch Lightning 完整指南

## 核心架构对比

| 原始实现 | Lightning 实现 | 说明 |
|---------|---------------|------|
| `forward_backward()` | `training_step()` | 训练逻辑主入口 |
| `backprop_loss()` | `manual_backward()` | 手动反向传播 |
| `update_ema()` | `on_train_batch_end()` | EMA 更新 hook |
| FSDP 手动设置 | `Trainer(strategy="fsdp")` | 分布式策略 |
| 自定义优化器 | `configure_optimizers()` | 优化器配置 |

---

## 步骤 1: 创建 LightningModule

```python
import pytorch_lightning as pl
import torch
from torch import nn

class DINOv3Lightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # ============ 模型初始化 ============
        self.student = nn.ModuleDict({
            "backbone": build_student_backbone(cfg),
            "dino_head": DINOHead(...),
            "ibot_head": DINOHead(...),
        })

        self.teacher = nn.ModuleDict({
            "backbone": build_teacher_backbone(cfg),
            "dino_head": DINOHead(...),
            "ibot_head": DINOHead(...),
        })
        self.teacher.requires_grad_(False)

        # ============ 损失函数 ============
        self.dino_loss = DINOLoss(cfg.dino.head_n_prototypes)
        self.ibot_loss = iBOTPatchLoss(cfg.ibot.head_n_prototypes)
        self.koleo_loss = KoLeoLoss()

        if cfg.gram.use_loss:
            self.gram_teacher = nn.ModuleDict({"backbone": ...})
            self.gram_loss = GramLoss(...)

        # ============ 关键设置 ============
        # 禁用自动优化（因为需要手动处理 EMA 更新）
        self.automatic_optimization = False

        # EMA 参数
        self.ema_momentum = 0.996

    def on_fit_start(self):
        """训练开始时初始化权重"""
        # 初始化 student
        self.student.backbone.init_weights()
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()

        # 复制 student 权重到 teacher
        self.teacher.load_state_dict(self.student.state_dict())
```

---

## 步骤 2: 实现训练逻辑

```python
    def training_step(self, batch, batch_idx):
        """
        对应原始代码的 forward_backward()
        """
        opt = self.optimizers()
        opt.zero_grad()

        # ============ 数据准备 ============
        global_crops = batch["collated_global_crops"]  # [2*B, 3, 224, 224]
        local_crops = batch["collated_local_crops"]    # [n_local*B, 3, 96, 96]
        masks = batch["collated_masks"]
        mask_indices_list = batch["mask_indices_list"]

        # ============ Teacher 前向传播 (无梯度) ============
        with torch.no_grad():
            teacher_out = self._get_teacher_output(
                global_crops,
                mask_indices_list=mask_indices_list,
                teacher_temp=self._get_teacher_temp(self.current_epoch)
            )

        # ============ Student 前向传播 ============
        student_global, student_local = self._get_student_output(
            global_crops=global_crops,
            local_crops=local_crops,
            masks=masks,
            mask_indices_list=mask_indices_list
        )

        # ============ 计算损失 ============
        losses = self._compute_losses(
            teacher_global=teacher_out,
            student_global=student_global,
            student_local=student_local,
            masks=masks,
            iteration=self.global_step
        )

        total_loss = losses["total"]

        # ============ 手动反向传播 ============
        self.manual_backward(total_loss)

        # ============ 梯度裁剪（可选）============
        if self.cfg.optim.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.cfg.optim.clip_grad
            )

        # ============ 优化器步进 ============
        opt.step()

        # ============ 记录指标 ============
        self.log_dict({
            "train/loss": total_loss,
            "train/dino_loss": losses["dino_loss"],
            "train/ibot_loss": losses["ibot_loss"],
            "train/koleo_loss": losses["koleo_loss"],
        }, sync_dist=True, prog_bar=True)

        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        对应原始代码的 update_ema()
        在每个 batch 结束后更新 Teacher EMA
        """
        # 计算 EMA 动量（可以根据 iteration 动态调整）
        m = self._get_ema_momentum(self.global_step)

        # 更新 teacher 参数
        with torch.no_grad():
            for param_student, param_teacher in zip(
                self.student.parameters(),
                self.teacher.parameters()
            ):
                param_teacher.data.mul_(m).add_(
                    param_student.data, alpha=1 - m
                )
```

---

## 步骤 3: 实现辅助方法

```python
    def _get_teacher_output(self, images, mask_indices_list, teacher_temp):
        """对应 get_teacher_output()"""
        n_crops, B = images.shape[0] // 2, images.shape[0] // 2
        images = images.flatten(0, 1)

        backbone_out = self.teacher.backbone(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]
        patch = backbone_out["x_norm_patchtokens"]

        # DINO head
        cls_after_head = self.teacher.dino_head(cls)
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head, teacher_temp=teacher_temp
        )

        # iBOT head (仅在 masked patches 上)
        masked_patches = torch.index_select(
            patch.flatten(0, 1), dim=0, index=mask_indices_list
        )
        masked_patch_after_head = self.teacher.ibot_head(masked_patches)
        masked_patch_centered = self.ibot_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head, teacher_temp=teacher_temp
        )

        return {
            "cls_centered": cls_centered.unflatten(0, (2, B)),
            "masked_patch_centered": masked_patch_centered,
            "patch_pre_head": patch.unflatten(0, (2, B)),
        }

    def _get_student_output(self, global_crops, local_crops, masks, mask_indices_list):
        """对应 get_student_output()"""
        # 联合前向传播全局和局部 crops
        global_out, local_out = self.student.backbone(
            [global_crops, local_crops],
            masks=[masks, None],
            is_training=True
        )

        g_cls = global_out["x_norm_clstoken"]
        g_patch = global_out["x_norm_patchtokens"]
        l_cls = local_out["x_norm_clstoken"]

        # DINO head on CLS tokens
        global_cls_after_head = self.student.dino_head(g_cls)
        local_cls_after_head = self.student.dino_head(l_cls)

        # iBOT head on masked patches
        masked_patches = torch.index_select(
            g_patch.flatten(0, 1), dim=0, index=mask_indices_list
        )
        masked_patch_after_head = self.student.ibot_head(masked_patches)

        return {
            "cls_after_head": global_cls_after_head,
            "masked_patch_after_head": masked_patch_after_head,
            "cls_pre_head": g_cls,
        }, {
            "cls_after_head": local_cls_after_head,
        }

    def _compute_losses(self, teacher_global, student_global,
                        student_local, masks, iteration):
        """对应 compute_losses()"""
        losses = {}

        # DINO loss (local crops)
        dino_local = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        losses["dino_local"] = dino_local * self.cfg.dino.loss_weight

        # DINO loss (global crops)
        dino_global = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=True
        )
        losses["dino_global"] = dino_global * self.cfg.dino.loss_weight

        # KoLeo loss
        koleo = self.koleo_loss(student_global["cls_pre_head"])
        losses["koleo"] = koleo * self.cfg.dino.koleo_loss_weight

        # iBOT loss
        ibot = self.ibot_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=masks.sum(),
        )
        losses["ibot"] = ibot * self.cfg.ibot.loss_weight

        # Total loss
        losses["total"] = sum(losses.values())

        return losses

    def _get_teacher_temp(self, epoch):
        """动态调整 teacher temperature"""
        warmup_teacher_temp = 0.04
        teacher_temp = 0.07
        warmup_teacher_temp_epochs = 30

        if epoch < warmup_teacher_temp_epochs:
            return warmup_teacher_temp + (teacher_temp - warmup_teacher_temp) * epoch / warmup_teacher_temp_epochs
        return teacher_temp

    def _get_ema_momentum(self, iteration):
        """动态调整 EMA 动量"""
        base_momentum = 0.996
        final_momentum = 1.0
        max_iterations = self.cfg.optim.epochs * self.cfg.train.OFFICIAL_EPOCH_LENGTH

        return base_momentum + (final_momentum - base_momentum) * iteration / max_iterations
```

---

## 步骤 4: 配置优化器和学习率调度器

```python
    def configure_optimizers(self):
        """
        对应原始代码的 get_params_groups() + optimizer setup
        """
        # 获取参数组（layer-wise lr decay）
        params_groups = self._get_params_groups_with_decay()

        # AdamW 优化器
        optimizer = torch.optim.AdamW(
            params_groups,
            lr=self.cfg.optim.base_lr,
            weight_decay=self.cfg.optim.weight_decay,
            betas=(0.9, 0.999),
        )

        # Cosine Annealing 学习率调度器
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.optim.epochs,
                eta_min=self.cfg.optim.min_lr,
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def _get_params_groups_with_decay(self):
        """
        Layer-wise learning rate decay
        类似 get_params_groups_with_decay_fsdp()
        """
        param_groups = []

        # Backbone parameters with layer-wise decay
        lr_scales = self._get_layer_lr_scales()
        for name, param in self.student.backbone.named_parameters():
            layer_id = self._get_layer_id(name)
            param_groups.append({
                "params": [param],
                "lr": self.cfg.optim.base_lr * lr_scales[layer_id],
                "weight_decay": self.cfg.optim.weight_decay,
            })

        # Head parameters (no decay)
        for head in ["dino_head", "ibot_head"]:
            param_groups.append({
                "params": self.student[head].parameters(),
                "lr": self.cfg.optim.base_lr,
                "weight_decay": 0.0,  # Heads typically use no weight decay
            })

        return param_groups
```

---

## 步骤 5: 创建 Trainer

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import FSDPStrategy

# ============ 回调函数 ============
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="dinov3-{epoch:02d}-{train/loss:.4f}",
    save_top_k=3,
    monitor="train/loss",
    mode="min",
    save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval="step")

# ============ FSDP 策略 ============
# 对应原始代码的 FSDP 配置
fsdp_strategy = FSDPStrategy(
    sharding_strategy="SHARD_GRAD_OP",  # 对应 cfg.compute_precision.sharding_strategy
    activation_checkpointing_policy={  # 对应 activation checkpointing
        torch.nn.TransformerEncoderLayer
    },
    cpu_offload=False,
)

# ============ Trainer ============
trainer = pl.Trainer(
    max_epochs=cfg.optim.epochs,
    accelerator="gpu",
    devices=8,  # 8 GPUs
    strategy=fsdp_strategy,
    precision="bf16-mixed",  # 对应 cfg.compute_precision
    callbacks=[checkpoint_callback, lr_monitor],
    gradient_clip_val=1.0 if cfg.optim.clip_grad else None,
    log_every_n_steps=10,
    accumulate_grad_batches=1,
    num_sanity_val_steps=0,  # 无验证集
)

# ============ 开始训练 ============
model = DINOv3Lightning(cfg)
trainer.fit(model, train_dataloader)
```

---

## 🚨 关键挑战和解决方案

### **挑战 1: 自定义分布式策略**
- **问题**: 原始代码使用自定义的 FSDP 设置（如 `ac_compile_parallelize`）
- **解决方案**:
  ```python
  # 方案 A: 使用 Lightning 的 FSDPStrategy（推荐）
  strategy = FSDPStrategy(
      sharding_strategy="SHARD_GRAD_OP",
      auto_wrap_policy=...,
  )

  # 方案 B: 自定义 Strategy（如果需要更多控制）
  from pytorch_lightning.strategies import Strategy
  class CustomFSDPStrategy(Strategy):
      def setup(self, trainer):
          # 调用原始的 ac_compile_parallelize
          ac_compile_parallelize(...)
  ```

### **挑战 2: Teacher EMA 更新时机**
- **问题**: EMA 需要在每个 batch 后更新，但 Lightning 默认在 optimizer.step() 后更新
- **解决方案**: 使用 `on_train_batch_end()` hook（已在上面实现）

### **挑战 3: 多个 Teacher 模型**
- **问题**: Gram teacher 需要单独的 EMA 更新逻辑
- **解决方案**:
  ```python
  def on_train_batch_end(self, outputs, batch, batch_idx):
      # 更新 main teacher
      self._update_ema(self.student, self.teacher, momentum=0.996)

      # 更新 gram teacher（如果需要）
      if self.cfg.gram.rep_update and \
         self.global_step % self.cfg.gram.update_frequency == 0:
          self._update_ema(self.teacher, self.gram_teacher, momentum=0.0)
  ```

### **挑战 4: Sinkhorn-Knopp Centering**
- **问题**: `dino_loss.sinkhorn_knopp_teacher()` 涉及跨 GPU 通信
- **解决方案**: Lightning 会自动处理 `all_gather`，但需要确保：
  ```python
  # 在 DINOLoss 中使用 Lightning 的分布式工具
  def sinkhorn_knopp_teacher(self, logits, teacher_temp):
      if self.trainer.world_size > 1:
          # Lightning 会自动注入 self.trainer
          gathered = self.all_gather(logits)  # 自动同步
      else:
          gathered = logits
      # ... Sinkhorn 算法
  ```

### **挑战 5: 数据加载器**
- **问题**: 原始代码使用自定义的 collate_fn 和 sampler
- **解决方案**:
  ```python
  class DINOv3DataModule(pl.LightningDataModule):
      def __init__(self, cfg):
          super().__init__()
          self.cfg = cfg

      def train_dataloader(self):
          dataset = ImageNetDataset(...)

          # 使用原始的 DataAugmentationDINO
          transform = DataAugmentationDINO(
              global_crops_scale=self.cfg.crops.global_crops_scale,
              local_crops_scale=self.cfg.crops.local_crops_scale,
              # ...
          )
          dataset.transform = transform

          return DataLoader(
              dataset,
              batch_size=self.cfg.train.batch_size_per_gpu,
              num_workers=self.cfg.train.num_workers,
              collate_fn=collate_data_and_cast,  # 自定义 collate
              shuffle=True,
              pin_memory=True,
          )
  ```

### **挑战 6: Checkpoint 兼容性**
- **问题**: 需要加载原始 DINOv3 的预训练权重
- **解决方案**:
  ```python
  def on_load_checkpoint(self, checkpoint):
      # 转换原始 checkpoint 格式
      if "student.backbone.pos_embed" in checkpoint["state_dict"]:
          # 原始格式
          new_state_dict = self._convert_legacy_checkpoint(
              checkpoint["state_dict"]
          )
          checkpoint["state_dict"] = new_state_dict
  ```

### **挑战 7: Mixed Precision Training**
- **问题**: 原始代码使用自定义的混合精度策略
- **解决方案**: Lightning 的 `precision="bf16-mixed"` 基本等价，但需要注意：
  ```python
  # 如果需要更细粒度的控制
  from pytorch_lightning.plugins import MixedPrecisionPlugin

  plugin = MixedPrecisionPlugin(
      precision="bf16-mixed",
      device="cuda",
      scaler=None,  # BF16 不需要 GradScaler
  )

  trainer = Trainer(plugins=[plugin])
  ```

---

## 📊 性能对比建议

| 方面 | 原始实现 | Lightning 实现 | 建议 |
|-----|---------|---------------|-----|
| **灵活性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 复杂自定义用原始 |
| **代码可读性** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Lightning 更清晰 |
| **训练速度** | 基准 | -5% ~ +2% | 性能相近 |
| **多 GPU 支持** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Lightning 更简单 |
| **调试难度** | ⭐⭐ | ⭐⭐⭐⭐ | Lightning 更友好 |

---

## ✅ 迁移检查清单

- [ ] **模型架构**: Student/Teacher/Gram teacher 都正确初始化
- [ ] **损失函数**: DINO/iBOT/KoLeo/Gram 都正确实现
- [ ] **EMA 更新**: Teacher 在每个 batch 后更新
- [ ] **分布式训练**: FSDP 配置正确
- [ ] **数据增强**: DataAugmentationDINO 保持不变
- [ ] **学习率调度**: Warmup + Cosine Annealing
- [ ] **梯度裁剪**: 如果需要
- [ ] **Checkpoint**: 能加载原始权重
- [ ] **日志记录**: WandB/TensorBoard 集成
- [ ] **验证**: 对比原始实现的 loss 曲线

---

## 🎯 最终建议

1. **适合 Lightning 的场景**:
   - 标准训练流程
   - 需要快速实验和调试
   - 多种分布式策略切换

2. **保持原始实现的场景**:
   - 高度自定义的 FSDP 配置
   - 复杂的通信模式
   - 极致性能优化

3. **混合方案**（推荐）:
   - 核心训练用 Lightning
   - 保留原始的数据增强和损失函数
   - 渐进式迁移，逐步测试
