# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DINOv3 fine-tuning project that implements a simplified version of the DINO (Self-Distillation with No Labels) self-supervised learning framework. The project includes:

1. **Mini-DINOv3**: A simplified educational implementation (~550 lines) focusing on core DINO concepts
2. **Full DINOv3**: The complete Meta AI DINOv3 implementation (in `dinov3/` directory) used as a backbone
3. **Training infrastructure**: PyTorch Lightning-based training with Hydra configuration

## Environment Setup

This project uses `uv` for Python environment management:

```bash
# Sync environment and install all dependencies
uv sync
```

The project requires Python >=3.12 and uses PyTorch with CUDA 12.8 support.

## Common Commands

### Testing

```bash
# Test core DINO components (loss, head, model)
python test_mini_dino.py

# Test specific components
python test/test_loss.py
python test/test_pca.py

# Run example demonstrations
python examples_mini_dino.py
```

### Training

```bash
# Train Mini-DINO model with custom dataset
python -m src.model.train_mini_dino \
    --data_dir /path/to/images \
    --output_dir outputs/mini_dino \
    --epochs 10 \
    --batch_size 8 \
    --lr 0.001

# Available backbone options: dinov3_vits14, dinov3_vitb14, dinov3_vitl14
# Adjust --out_dim for number of prototypes (default: 8192)
```

### Configuration

The project uses Hydra for configuration management. Config files are in `config/`:

- `config/train.yaml` - Main training configuration
- `config/model/` - Model architectures (e.g., dinov3_vits16.yaml)
- `config/loss/` - Loss functions (dino.yaml, ibot.yaml, gram.yaml)
- `config/data/` - Dataset configurations

## Code Architecture

### Directory Structure

```
src/
├── model/           # Model implementations
│   ├── ssl_ref.py          # MiniDINO and DINOHead classes (main implementation)
│   ├── loss.py             # DINOLoss with Sinkhorn-Knopp algorithm
│   ├── train_mini_dino.py  # Standalone training script
│   ├── head.py             # Projection head
│   └── dinov3_lightning.py # PyTorch Lightning wrapper
├── data/            # Data loading utilities
│   └── test_datamodule.py  # Simple image loader
└── utils/           # Utility functions
    ├── visualize.py        # PCA feature visualization
    └── image_process.py    # Image transformations

dinov3/              # Full DINOv3 implementation (Meta AI)
├── dinov3/          # Core DINOv3 package
│   ├── models/      # Vision Transformer architectures
│   ├── layers/      # Custom layers (attention, FFN, etc.)
│   ├── loss/        # Loss functions (DINO, KoLeo, iBOT, Gram)
│   ├── data/        # Data augmentation and loading
│   ├── train/       # Training infrastructure
│   └── eval/        # Evaluation tasks (segmentation, depth, etc.)
└── hubconf.py       # torch.hub integration

config/              # Hydra configuration files
test/                # Unit tests
weights/             # Pretrained model weights
reference/           # Reference implementations and experiments
```

### Key Components

#### 1. MiniDINO Model ([src/model/ssl_ref.py](src/model/ssl_ref.py))

The core self-distillation model with student-teacher architecture:

- **Student network**: Trained with gradients, learns from teacher
- **Teacher network**: EMA of student, provides stable targets (no gradients)
- **DINOHead**: 3-layer MLP projection head (in_dim → hidden_dim → bottleneck_dim → out_dim)

Key methods:
- `forward()`: Computes student/teacher outputs and DINO loss
- `update_teacher(momentum=0.996)`: EMA update of teacher parameters
- `get_student_parameters()`: Returns only student parameters for optimization

#### 2. DINO Loss ([src/model/loss.py](src/model/loss.py))

Implements the DINO self-supervised loss:

- **Sinkhorn-Knopp algorithm**: Prevents mode collapse by ensuring uniform prototype usage
- **Center update**: EMA-based centering to stabilize training
- **Temperature scaling**: Student temp (0.1) vs Teacher temp (0.04)

Key methods:
- `sinkhorn_knopp_teacher()`: Normalizes teacher outputs to doubly stochastic matrix
- `forward()`: Computes cross-entropy loss between student and teacher
- `update_center()`: Updates running center with EMA

#### 3. DINOv3 Backbone Integration

The project uses pretrained DINOv3 backbones via `torch.hub.load()`:

```python
# Load from local dinov3 directory
backbone = torch.hub.load(
    repo_or_dir='dinov3',
    model='dinov3_vits16',
    source='local',
    weights='weights/dinov3_vits16_pretrain.pth'
)
```

Available backbones:
- `dinov3_vits14` / `dinov3_vits16`: ViT-Small (384-dim features, 21M params)
- `dinov3_vitb14` / `dinov3_vitb16`: ViT-Base (768-dim features, 86M params)
- `dinov3_vitl14` / `dinov3_vitl16`: ViT-Large (1024-dim features, 300M params)

Backbone outputs:
- `x_norm_clstoken`: Global image features [B, D]
- `x_norm_patchtokens`: Local patch features [B, N_patches, D]

### Training Flow

1. **Data augmentation**: RandomResizedCrop, ColorJitter, GaussianBlur, normalization
2. **Student forward**: Extract features → Project to prototype space
3. **Teacher forward** (no_grad): Extract features → Project to prototype space
4. **Sinkhorn-Knopp**: Normalize teacher outputs to prevent collapse
5. **Loss computation**: Cross-entropy between student logits and teacher probs
6. **Backward pass**: Only student parameters receive gradients
7. **Teacher update**: EMA update with momentum 0.996
8. **Center update**: Update running center for next iteration

### Important Implementation Details

#### Teacher Network Behavior

- Teacher is initialized as a copy of student
- Teacher parameters are frozen (no gradients)
- Teacher is updated via EMA: `teacher = 0.996 * teacher + 0.004 * student`
- This provides stable, slowly-evolving targets for the student

#### Sinkhorn-Knopp Algorithm

Prevents mode collapse by ensuring:
1. Each prototype is used equally across the batch (row normalization)
2. Each sample distributes weight equally across prototypes (column normalization)
3. Iterative refinement (typically 3 iterations)

#### Numerical Stability

The implementation includes safeguards:
- Epsilon (1e-8) added to prevent division by zero
- Log-sum-exp trick for numerical stability in softmax
- Gradient clipping (max_norm=3.0) during training

## Development Notes

### Model Weights

Pretrained DINOv3 weights should be placed in `weights/` directory:
- Download from Meta AI's DINOv3 release
- Use `wget` instead of browser for downloading
- Configure path in `config/model/*.yaml` files

### Feature Extraction

To extract features from trained models:

```python
from src.model.ssl_ref import MiniDINO

model = MiniDINO(backbone_name='dinov3_vits14', out_dim=8192)
model.eval()

with torch.no_grad():
    # Global features (CLS token)
    cls_features = model.student_backbone(images)['x_norm_clstoken']  # [B, 384]

    # Local features (patch tokens)
    patch_features = model.student_backbone(images)['x_norm_patchtokens']  # [B, 256, 384]

    # Prototype logits
    logits = model.student_head(cls_features)  # [B, 8192]
```

### Visualization

Use PCA to visualize patch features:

```python
from src.utils.visualize import pca_transform_features

# patch_features: [B, N_patches, D]
pca_images = pca_transform_features(patch_features, patch_h=16, patch_w=16, n_components=3)
# Returns: [B, H, W, 3] RGB visualization
```

### Configuration with Hydra

Override config values from command line:

```bash
# Change model backbone
python train.py model=dinov3_vitb16

# Change loss parameters
python train.py loss.student_temp=0.15 loss.center_momentum=0.95

# Change data path
python train.py data.path=/path/to/data
```

## Differences from Full DINOv3

Mini-DINOv3 is simplified for educational purposes:

| Feature | Full DINOv3 | Mini-DINOv3 |
|---------|-------------|-------------|
| Multi-crop | 2 global + 8 local crops | 1 global crop |
| Loss functions | DINO + KoLeo + iBOT + Gram | DINO only |
| Distributed training | Multi-GPU with FSDP | Single GPU |
| Backbone | Custom ViT with registers | Pretrained DINOv3 ViT |
| Code complexity | ~10,000 lines | ~550 lines |

For production use or research, refer to the full implementation in `dinov3/`.

## Troubleshooting

### Loss becomes NaN

- Check Sinkhorn-Knopp numerical stability (epsilon values)
- Reduce learning rate
- Check for inf/nan in input images

### Out of memory

- Reduce `batch_size`
- Reduce `out_dim` (number of prototypes)
- Use smaller backbone (vits14 instead of vitb14)
- Enable gradient checkpointing

### Training too slow

- Use smaller dataset for testing
- Reduce number of epochs
- Use smaller backbone
- Ensure CUDA is available and being used

### Model download fails

- Check network connection
- Verify `dinov3/` directory exists and contains `hubconf.py`
- Check weights path in config files
- Use local weights instead of downloading from URL
