# DINOv3 Fine-tuning Framework

<div align="center">

**一个简洁、易懂、功能完整的 DINOv3 自监督学习实现**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[特性](#-特性) •
[快速开始](#-快速开始) •
[文档](#-文档) •
[实验结果](#-实验结果) •
[贡献](#-贡献)

</div>

---

## 📖 简介

这是一个**教育友好**的 DINOv3 实现，包含了所有核心算法，但代码更简洁（~730 行 vs 官方 ~10,000 行）。非常适合：

- 🎓 **学习自监督学习**：清晰的代码结构和详细的注释
- 🔬 **研究实验**：模块化设计，易于修改和扩展
- 🚀 **实际应用**：功能完整，可直接用于训练

### 核心特性

- ✅ **Sinkhorn-Knopp 归一化**：防止模型坍塌
- ✅ **EMA 教师更新**：稳定的训练动态
- ✅ **Multi-crop 增强**：2 global + 8 local crops
- ✅ **完整的损失函数**：DINO + iBOT + KoLeo + Gram
- ✅ **丰富的文档**：4 个详细文档 + 代码注释

---

## 🎯 特性

### 核心算法

| 功能 | 状态 | 说明 |
|------|------|------|
| Sinkhorn-Knopp 归一化 | ✅ | 防止模式坍塌，确保原型均匀使用 |
| EMA 教师更新 | ✅ | 指数移动平均，提供稳定训练目标 |
| Multi-crop 数据增强 | ✅ | 2 global + 8 local crops |
| DINO Loss | ✅ | CLS token 自蒸馏损失 |
| iBOT Loss | ✅ | Patch-level 自监督损失 |
| KoLeo Loss | ✅ | 熵正则化损失 |
| Gram Loss | ✅ | 特征相关性损失 |

### 训练功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 学习率调度 | ✅ | Warmup + Cosine decay |
| 梯度裁剪 | ✅ | 防止梯度爆炸 |
| Checkpoint 保存/加载 | ✅ | 支持训练恢复 |
| 日志记录 | ✅ | JSON Lines 格式 |
| 配置管理 | ✅ | Hydra 配置系统 |

---

## 🚀 快速开始

### 环境要求

- Python >= 3.12
- PyTorch >= 2.0
- CUDA >= 12.8 (推荐)

### 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd DINOv3-finetune

# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 快速测试

```bash
# 1. 测试 Sinkhorn-Knopp + EMA（1步训练）
python test/test_loss.py

# 2. EMA 详细演示（5步训练）
python test/test_ema_update.py

# 3. EMA 可视化（20步训练 + 图表）
python test/visualize_ema.py
```

**预期输出**:
```
✓ Teacher outputs (with Sinkhorn-Knopp centering):
  global_dino_centered: torch.Size([8, 65536])
  local_dino_centered: torch.Size([32, 65536])
✓ Loss: 33.1093
✓ Teacher updated with EMA (momentum=0.996)
```

### 开始训练

```bash
# 使用默认配置训练
python train_complete.py

# 自定义配置
python train_complete.py \
    data.data_path=data/my_dataset \
    train.epochs=100 \
    train.lr=0.001 \
    output_dir=outputs/my_experiment

# 从 checkpoint 恢复
python train_complete.py \
    resume_from=outputs/dinov3_training/checkpoint_latest.pth
```

---

## 📚 文档

### 核心文档

| 文档 | 内容 | 适合 |
|------|------|------|
| [EMA 教师更新详解](docs/EMA_TEACHER_UPDATE.md) | EMA 原理、数学推导、实现细节 | 深入理解 EMA |
| [实现总结](docs/IMPLEMENTATION_SUMMARY.md) | 完整实现、代码架构、实验结果 | 了解整体实现 |
| [快速参考](docs/QUICK_REFERENCE.md) | 常用命令、配置、问题排查 | 日常使用 |
| [训练指南](docs/TRAINING_GUIDE.md) | 完整训练流程、技巧、实验建议 | 开始训练 |

### 推荐阅读顺序

**初学者**:
1. 本 README
2. 运行 `python test/test_loss.py`
3. [EMA 教师更新详解](docs/EMA_TEACHER_UPDATE.md)
4. [快速参考](docs/QUICK_REFERENCE.md)

**进阶**:
1. [实现总结](docs/IMPLEMENTATION_SUMMARY.md)
2. [训练指南](docs/TRAINING_GUIDE.md)
3. 修改代码，进行实验

---

## 📊 实验结果

### Sinkhorn-Knopp 归一化

```python
# 测试命令
python test/test_loss.py

# 结果
✓ 所有输出成功归一化
✓ 损失值稳定（~33.1）
✓ 没有 NaN 或 Inf
```

### EMA 教师更新

```python
# 测试命令
python test/test_ema_update.py

# 结果
Step 1/5: Teacher moved 0.40% closer to student ✓
Step 5/5: Teacher moved 0.40% closer to student ✓
Average Movement Ratio: 0.4016% (Expected: 0.4000%) ✓
```

### 可视化分析

```python
# 测试命令
python test/visualize_ema.py

# 生成图表
outputs/ema_dynamics.png
```

**关键发现**:
- ✅ 损失逐渐下降（训练有效）
- ✅ 学生-教师距离增长（学生在学习）
- ✅ 参数范数稳定（无梯度爆炸/消失）
- ✅ EMA 移动比例与理论值完全一致

---

## 🏗️ 项目结构

```
DINOv3-finetune/
│
├── src/                          # 核心实现
│   ├── model/
│   │   ├── ssl.py               ⭐ Student/Teacher + Sinkhorn-Knopp
│   │   ├── ssl_ref.py           # MiniDINO 参考实现
│   │   ├── head.py              # DINO Head
│   │   └── data.py              # Multi-crop dataloader
│   ├── loss/
│   │   ├── dino_clstoken_loss.py  # DINO Loss
│   │   ├── ibot_patch_loss.py     # iBOT Loss
│   │   ├── koleo_loss.py          # KoLeo Loss
│   │   └── gram_loss.py           # Gram Loss
│   └── utils/
│       └── visualize.py         # PCA 可视化
│
├── test/                         # 测试脚本
│   ├── test_loss.py             ⭐ 基础测试
│   ├── test_ema_update.py       ⭐ EMA 演示
│   └── visualize_ema.py         ⭐ EMA 可视化
│
├── config/                       # 配置文件
│   ├── train.yaml               # 主配置
│   ├── train_complete.yaml      # 完整训练配置
│   ├── model/                   # 模型配置
│   ├── loss/                    # 损失配置
│   └── data/                    # 数据配置
│
├── docs/                         # 文档
│   ├── EMA_TEACHER_UPDATE.md    ⭐ EMA 详解
│   ├── IMPLEMENTATION_SUMMARY.md ⭐ 实现总结
│   ├── QUICK_REFERENCE.md       ⭐ 快速参考
│   ├── TRAINING_GUIDE.md        ⭐ 训练指南
│   └── PROJECT_COMPLETION.md    # 项目完成总结
│
├── train_complete.py            ⭐ 完整训练脚本
├── CLAUDE.md                    # 项目说明
└── README.md                    # 本文件
```

---

## 🔬 核心算法

### 1. Sinkhorn-Knopp 归一化

**作用**: 防止模型坍塌，确保所有原型被均匀使用

```python
class SinkhornKnoppCentering(nn.Module):
    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        Q = torch.exp(teacher_output / teacher_temp).t()
        B = n_masked_patches_tensor
        K = Q.shape[0]

        Q /= torch.sum(Q)

        for _ in range(n_iterations):
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()
```

**关键点**:
- 迭代 3 次进行双随机归一化
- 确保每行和 = 1/K，每列和 = 1/B
- 提供稳定的训练目标

### 2. EMA 教师更新

**作用**: 教师参数是学生历史参数的指数加权平均

```python
@torch.no_grad()
def update_teacher_ema(student, teacher, momentum=0.996):
    """
    Formula: teacher_param = momentum * teacher_param + (1 - momentum) * student_param
    """
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)
```

**关键点**:
- 动量 = 0.996（标准设置）
- 每步教师向学生移动 0.4%
- 有效历史窗口约 250 步

---

## 🎓 学习资源

### 论文

1. **DINO (2021)**: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
2. **DINOv2 (2023)**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
3. **iBOT (2022)**: [iBOT: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/abs/2111.07832)
4. **Mean Teacher (2017)**: [Mean teachers are better role models](https://arxiv.org/abs/1703.01780)

### 官方实现

- [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) - 官方 DINOv3 实现

---

## 🛠️ 常见问题

### Q1: Loss 变成 NaN 怎么办？

**解决方案**:
```bash
# 降低学习率
python train_complete.py train.lr=0.0001

# 增加梯度裁剪
python train_complete.py train.grad_clip=1.0
```

### Q2: GPU 内存不足怎么办？

**解决方案**:
```bash
# 减小 batch size 和原型数量
python train_complete.py \
    data.batch_size=2 \
    model.dino_head.out_dim=8192 \
    data.num_local_crops=4
```

### Q3: 训练太慢怎么办？

**解决方案**:
```bash
# 增加 num_workers
python train_complete.py data.num_workers=8

# 减少日志频率
python train_complete.py train.log_interval=100
```

更多问题请查看 [快速参考](docs/QUICK_REFERENCE.md)。

---

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [Meta AI](https://ai.facebook.com/) - 原始 DINOv3 实现
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Hydra](https://hydra.cc/) - 配置管理

---

## 🌟 Star History

如果这个项目对你有帮助，请给个 ⭐️！

---

<div align="center">

**Made with ❤️ for Machine Learning Research**

[⬆ 回到顶部](#dinov3-fine-tuning-framework)

</div>
