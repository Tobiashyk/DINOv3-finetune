# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DINOv3 fine-tuning project that implements a modular version of the DINO (Self-Distillation with No Labels) self-supervised learning framework. The project includes:

1. **Modular DINOv3**: A clean, educational implementation (~730 lines) with all major loss functions
2. **Full DINOv3**: The complete Meta AI DINOv3 implementation (in `dinov3/` directory) used as a backbone
3. **Training infrastructure**: PyTorch Lightning-based training with Hydra configuration
4. **Multiple loss functions**: DINO, KoLeo, iBOT, and Gram losses for comprehensive self-supervised learning

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

- `config/train.yaml` - Main training configuration with defaults
- `config/model/` - Model architectures and backbone configs
  - `dinov3_vits16.yaml` - ViT-Small/16 with 384-dim features
- `config/loss/` - Loss function configurations
  - `dino.yaml` - DINO loss with Sinkhorn-Knopp (out_dim: 65536)
  - `ibot.yaml` - iBOT patch-level loss
  - `gram.yaml` - Gram matrix correlation loss
- `config/data/` - Dataset and augmentation configs
  - `test_dataset.yaml` - Multi-crop settings (2 global + 8 local crops)
  - `transform.yaml` - Augmentation parameters

Example `config/train.yaml`:
```yaml
defaults:
  - model: dinov3_vits16
  - loss: dino
  - data: test_dataset
  - _self_

train:
  epochs: 10
  lr: 1e-3
```

Example `config/loss/dino.yaml`:
```yaml
_target_: src.loss.dino_clstoken_loss.DINOLoss
out_dim: 65536
student_temp: 0.1
center_momentum: 0.9
```

## Code Architecture

### Directory Structure

```
src/
├── loss/            # Loss function implementations
│   ├── dino_clstoken_loss.py  # DINO loss for CLS tokens with Sinkhorn-Knopp
│   ├── koleo_loss.py          # KoLeo entropic regularization loss
│   ├── ibot_patch_loss.py     # iBOT patch-level self-supervised loss
│   └── gram_loss.py           # Gram matrix correlation loss
├── model/           # Model implementations
│   ├── ssl_ref.py          # MiniDINO and DINOHead classes (main implementation)
│   ├── ssl.py              # StudentSSLModel and TeacherSSLModel wrappers
│   ├── loss.py             # Legacy DINOLoss (for reference)
│   ├── head.py             # DINOHead projection head
│   ├── data.py             # SimpleImageDataset and dataloader utilities
│   ├── train_mini_dino.py  # Standalone training script
│   └── dinov3_lightning.py # PyTorch Lightning wrapper
└── utils/           # Utility functions
    ├── visualize.py        # PCA feature visualization
    └── image_process.py    # Multi-crop augmentation transforms

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
├── train.yaml       # Main training config
├── model/           # Model configs (dinov3_vits16.yaml, etc.)
├── loss/            # Loss configs (dino.yaml, ibot.yaml, gram.yaml)
└── data/            # Dataset configs (test_dataset.yaml, transform.yaml)
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

#### 2. Loss Functions

The project implements multiple self-supervised loss functions in `src/loss/`:

##### DINO Loss ([src/loss/dino_clstoken_loss.py](src/loss/dino_clstoken_loss.py))

Implements the DINO self-supervised loss for CLS tokens:

- **Sinkhorn-Knopp algorithm**: Prevents mode collapse by ensuring uniform prototype usage
- **Center update**: EMA-based centering to stabilize training
- **Temperature scaling**: Student temp (0.1) vs Teacher temp (0.04)
- **Multi-crop support**: Handles multiple global and local crops with optional diagonal masking

Key methods:
- `sinkhorn_knopp_teacher()`: Normalizes teacher outputs to doubly stochastic matrix
- `softmax_center_teacher()`: Alternative centering with softmax
- `forward()`: Computes cross-entropy loss between student and teacher crops
- `update_center()`: Updates running center with EMA

##### KoLeo Loss ([src/loss/koleo_loss.py](src/loss/koleo_loss.py))

Kozachenko-Leonenko entropic regularization loss:

- Encourages uniform distribution of features in embedding space
- Prevents feature collapse by maximizing nearest-neighbor distances
- Uses L2-normalized features and pairwise distance computation

##### iBOT Loss ([src/loss/ibot_patch_loss.py](src/loss/ibot_patch_loss.py))

Image BERT-style patch-level self-supervised loss:

- Operates on patch tokens instead of CLS tokens
- Uses masked patch prediction similar to BERT
- Includes Sinkhorn-Knopp normalization for patch-level features
- Supports both full and masked forward passes

##### Gram Loss ([src/loss/gram_loss.py](src/loss/gram_loss.py))

Gram matrix correlation loss:

- Computes MSE between Gram matrices of student and teacher features
- Captures feature correlations and relationships
- Supports image-level or batch-level computation
- Optional negative value removal for stability

#### 3. DINOHead Projection ([src/model/head.py](src/model/head.py))

3-layer MLP projection head that maps backbone features to prototype space:

- Architecture: `in_dim → hidden_dim → hidden_dim → bottleneck_dim → out_dim`
- L2 normalization before final projection
- Optional batch normalization
- Truncated normal weight initialization

#### 4. Data Loading ([src/model/data.py](src/model/data.py))

Multi-crop data augmentation pipeline:

- **Global crops**: 2 crops with different augmentation strategies
  - Crop 1: Strong Gaussian blur (p=1.0)
  - Crop 2: Weak Gaussian blur (p=0.1) + solarization
- **Local crops**: 8 smaller crops with moderate augmentation
- **Augmentations**: RandomResizedCrop, ColorJitter, GaussianBlur, normalization
- **SimpleImageDataset**: Loads images from directory and applies transforms

#### 5. DINOv3 Backbone Integration

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

1. **Data augmentation**: Multi-crop strategy
   - 2 global crops (224×224) with different augmentation strengths
   - 8 local crops (96×96) with moderate augmentation
   - Color jittering, Gaussian blur, random grayscale, solarization
   - ImageNet normalization

2. **Student forward**: For each crop
   - Extract features from backbone (CLS token + patch tokens)
   - Project CLS token through DINOHead to prototype space
   - Optionally project patch tokens for iBOT loss

3. **Teacher forward** (no_grad): For each crop
   - Extract features from teacher backbone
   - Project through teacher head
   - Apply centering to stabilize training

4. **Loss computation**:
   - **DINO loss**: Sinkhorn-Knopp normalization + cross-entropy between all crop pairs
   - **KoLeo loss** (optional): Entropic regularization on student features
   - **iBOT loss** (optional): Masked patch prediction
   - **Gram loss** (optional): Feature correlation matching

5. **Backward pass**: Only student parameters receive gradients

6. **Teacher update**: EMA update with momentum 0.996
   - `teacher = 0.996 * teacher + 0.004 * student`

7. **Center update**: Update running center for next iteration
   - `center = 0.9 * center + 0.1 * batch_mean`

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

The algorithm transforms teacher outputs into a doubly stochastic matrix:
```python
Q = exp(teacher_output / teacher_temp).T  # [K, B]
Q /= sum(Q)  # Normalize to sum to 1

for _ in range(3):
    Q /= sum(Q, dim=1, keepdim=True)  # Row normalization
    Q /= K
    Q /= sum(Q, dim=0, keepdim=True)  # Column normalization
    Q /= B

Q *= B  # Scale so columns sum to 1
return Q.T  # [B, K]
```

#### Multi-Crop Strategy

The implementation uses a multi-crop strategy to improve feature learning:

- **Global crops** (2 crops, 224×224):
  - Cover large portions of the image (scale: 0.32-1.0)
  - Crop 1: Strong Gaussian blur (p=1.0) for consistency
  - Crop 2: Weak blur (p=0.1) + solarization for diversity

- **Local crops** (8 crops, 96×96):
  - Cover small portions of the image (scale: 0.05-0.32)
  - Moderate Gaussian blur (p=0.5)
  - Encourage learning of local features

All crops are processed through both student and teacher networks, and the loss is computed between all student-teacher crop pairs (with optional diagonal masking to exclude same-crop comparisons).

#### Numerical Stability

The implementation includes safeguards:
- Epsilon (1e-8) added to prevent division by zero
- Log-sum-exp trick for numerical stability in softmax
- Gradient clipping (max_norm=3.0) during training

## Development Notes

### Loss Function Selection

The project supports multiple loss functions that can be combined:

**DINO Loss** (`src/loss/dino_clstoken_loss.py`):
- Primary self-distillation loss for CLS tokens
- Uses Sinkhorn-Knopp to prevent collapse
- Best for learning global image representations
- Config: `config/loss/dino.yaml`

**KoLeo Loss** (`src/loss/koleo_loss.py`):
- Entropic regularization to spread features uniformly
- Prevents feature collapse by maximizing nearest-neighbor distances
- Typically used as an auxiliary loss with weight 0.1
- No configuration needed (no hyperparameters)

**iBOT Loss** (`src/loss/ibot_patch_loss.py`):
- Patch-level self-supervised learning
- Masked patch prediction similar to BERT
- Better for dense prediction tasks (segmentation, detection)
- Config: `config/loss/ibot.yaml`

**Gram Loss** (`src/loss/gram_loss.py`):
- Matches feature correlations between student and teacher
- Captures higher-order feature relationships
- Useful for style transfer and texture learning
- Config: `config/loss/gram.yaml`

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

### Using Individual Loss Functions

You can use loss functions independently:

```python
from src.loss.dino_clstoken_loss import DINOLoss
from src.loss.koleo_loss import KoLeoLoss
from src.loss.ibot_patch_loss import iBOTPatchLoss
from src.loss.gram_loss import GramLoss

# DINO loss for CLS tokens
dino_loss = DINOLoss(out_dim=65536, student_temp=0.1, center_momentum=0.9)
dino_loss.init_weights()

# Prepare teacher outputs with Sinkhorn-Knopp
teacher_probs = dino_loss.sinkhorn_knopp_teacher(teacher_logits, teacher_temp=0.04)

# Compute loss (supports multi-crop)
loss = dino_loss(student_logits, teacher_probs, ignore_diagonal=True)
dino_loss.update_center(teacher_logits)

# KoLeo loss for feature regularization
koleo_loss = KoLeoLoss()
koleo = koleo_loss(student_features)  # [B, D] normalized features

# iBOT loss for patch tokens
ibot_loss = iBOTPatchLoss(patch_out_dim=8192, student_temp=0.1)
ibot_loss.init_weights()

# Prepare teacher patch outputs
teacher_patch_probs = ibot_loss.sinkhorn_knopp_teacher(
    teacher_patch_logits, teacher_temp=0.04, n_masked_patches_tensor=n_masked
)

# Compute masked patch loss
loss = ibot_loss.forward_masked(
    student_patch_tokens_masked, teacher_patch_probs, student_masks_flat
)

# Gram loss for feature correlations
gram_loss = GramLoss(apply_norm=True, img_level=True)
loss = gram_loss(student_patch_tokens, teacher_patch_tokens, img_level=True)
```

### Visualization

Use PCA to visualize patch features:

```python
from src.utils.visualize import pca_transform_features

# patch_features: [B, N_patches, D]
pca_images = pca_transform_features(patch_features, patch_h=16, patch_w=16, n_components=3)
# Returns: [B, H, W, 3] RGB visualization
```

### Multi-Crop Data Loading

The data loading pipeline supports flexible multi-crop configurations:

```python
from src.model.data import gen_dataloader
from omegaconf import DictConfig

# Configure multi-crop settings
data_cfg = DictConfig({
    'data_path': 'data/images',
    'batch_size': 4,
    'num_workers': 4,
    'global_crop_size': [224, 224],
    'global_crop_scale': [0.32, 1.0],
    'local_crop_size': [96, 96],
    'local_crop_scale': [0.05, 0.32],
    'num_local_crops': 8,
    'horizontal_flips': True
})

dataloader = gen_dataloader(data_cfg)

# Each batch returns a list of crops:
# [global_crop_1, global_crop_2, local_crop_1, ..., local_crop_8]
for crops in dataloader:
    # crops is a list of 10 tensors (2 global + 8 local)
    global_crops = crops[:2]  # [2, B, 3, 224, 224]
    local_crops = crops[2:]   # [8, B, 3, 96, 96]
```

### Configuration with Hydra

Override config values from command line:

```bash
# Change model backbone
python train.py model=dinov3_vitb16

# Change loss function
python train.py loss=ibot  # Use iBOT loss instead of DINO

# Change loss parameters
python train.py loss.student_temp=0.15 loss.center_momentum=0.95

# Change data path and batch size
python train.py data.data_path=/path/to/data data.batch_size=16

# Change multi-crop settings
python train.py data.num_local_crops=10 data.local_crop_size=[80,80]

# Combine multiple overrides
python train.py model=dinov3_vitb16 loss=dino data.batch_size=8 train.lr=5e-4
```

## Differences from Full DINOv3

The current implementation bridges Mini-DINOv3 and Full DINOv3:

| Feature | Full DINOv3 | Current Implementation |
|---------|-------------|------------------------|
| Multi-crop | 2 global + 8 local crops | ✅ 2 global + 8 local crops |
| Loss functions | DINO + KoLeo + iBOT + Gram | ✅ All 4 losses implemented |
| Distributed training | Multi-GPU with FSDP | ❌ Single GPU |
| Backbone | Custom ViT with registers | Pretrained DINOv3 ViT |
| Code complexity | ~10,000 lines | ~730 lines (modular) |
| Sinkhorn-Knopp | Compiled version | Standard PyTorch |
| Masking strategy | Advanced masking | Basic masking |

**Key advantages of current implementation:**
- Modular loss functions in separate files for easy experimentation
- Cleaner code structure with clear separation of concerns
- Educational value while maintaining production-ready features
- Easy to extend with custom losses or backbones

For distributed training or advanced features, refer to the full implementation in `dinov3/`.

## Troubleshooting

### Loss becomes NaN

- Check Sinkhorn-Knopp numerical stability (epsilon values)
- Reduce learning rate (try 1e-4 instead of 1e-3)
- Check for inf/nan in input images
- Verify center initialization: call `loss.init_weights()` before training
- Reduce number of Sinkhorn-Knopp iterations from 3 to 2

### Out of memory

- Reduce `batch_size` (try 2 or 4)
- Reduce `out_dim` (number of prototypes, try 8192 instead of 65536)
- Reduce number of local crops (try 4 instead of 8)
- Use smaller backbone (vits14 instead of vitb14)
- Enable gradient checkpointing
- Use mixed precision training (fp16)

### Training too slow

- Use smaller dataset for testing
- Reduce number of epochs
- Reduce number of local crops
- Use smaller backbone
- Ensure CUDA is available and being used
- Increase `num_workers` in dataloader (try 8 or 16)
- Use smaller prototype dimension (out_dim)

### Model download fails

- Check network connection
- Verify `dinov3/` directory exists and contains `hubconf.py`
- Check weights path in config files (e.g., `weights/dinov3_vits16_pretrain.pth`)
- Use local weights instead of downloading from URL
- For torch.hub.load, use `source='local'` parameter

### Multi-crop issues

- Verify crop sizes are compatible with backbone patch size (16 for vits16)
- Check that global_crop_scale and local_crop_scale don't overlap too much
- Ensure local crops are smaller than global crops
- If using ignore_diagonal=True, verify student and teacher have same number of crops

### Loss function compatibility

- DINO loss requires CLS token outputs (shape: [crops, B, out_dim])
- iBOT loss requires patch token outputs (shape: [B, N_patches, patch_out_dim])
- KoLeo loss requires normalized features (shape: [B, D])
- Gram loss requires patch tokens (shape: [B, N_patches, D])
- Make sure to call `init_weights()` on loss modules before training
