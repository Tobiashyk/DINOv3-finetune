"""
Self-Attention Visualization Script for DINOv3 with LoRA Fine-tuning

This script visualizes the [CLS] token's self-attention map from the last Transformer block
of DINOv3 models (base and LoRA-finetuned), allowing comparison of global attention patterns.

The attention map shows which spatial regions the [CLS] token focuses on, reflecting
the model's global understanding of the image.

Usage:
    python visualize_attention.py

To modify settings, edit the CONFIG section below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from peft import PeftModel
from typing import Dict, Tuple, Optional

# ==================== CONFIG ====================
# 修改以下配置参数以适应你的需求

# 1. 模型路径配置
BASE_WEIGHTS_PATH = '../../weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
LORA_WEIGHTS = 'new_version2_dinoloss_new_version2_ibotloss/trainall_C_and_MoS2_512_batch16_noisyall'  # 要改的
LORA_WEIGHTS_DIR = '../../student_weights/' + LORA_WEIGHTS
LORA_EPOCH = '6'  # 要改的，确保与训练时保存的epoch一致
MODEL_NAME = 'dinov3_vits16plus'  # 可选: dinov3_vits16plus, dinov3_vits16, dinov3_vitb16, etc.
size = 1024

# 2. 输入输出路径配置
INPUT_IMAGE = 'graphene_AA_test_only5'  # 要改的
# INPUT_IMAGE = 'MoS2_2H_test_only5'  # 要改的
# INPUT_IMAGE_DIR = '../../data/test_pic/' + INPUT_IMAGE
INPUT_IMAGE_DIR = '../../data/koala_test/'
OUTPUT_DIR = '../../data/visualize_attention/' + 'koala' + '/' + LORA_WEIGHTS + '/' + 'weights_epoch' + LORA_EPOCH + '/' + 'size' + str(size)
# OUTPUT_DIR = '../../data/visualize_attention/' + INPUT_IMAGE + '/' + LORA_WEIGHTS + '/' + 'weights_epoch' + LORA_EPOCH + '/' + 'size' + str(size)
IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp']  # 支持的图像格式

# 3. 图像处理配置
SCALE_FACTOR = 1                    # 图像放大倍数 (1=原始大小, 2=放大2倍, etc.)
PATCH_SIZE = 16                     # 模型 patch size (通常为16)
TARGET_SIZE = (size, size)          # 图像预处理目标尺寸 (H, W)
RESIZE_SCALE = (1.0, 1.0)           # RandomResizedCrop scale
RESIZE_RATIO = (1.0, 1.0)           # RandomResizedCrop aspect ratio

# 4. 注意力可视化配置
COLORMAP_ATTENTION = 'jet'          # 注意力热力图颜色映射: 'jet', 'inferno', 'hot', 'viridis'
OVERLAY_ALPHA = 0.6                 # 叠加图透明度 (0-1, 越大越不透明)
OUTPUT_DPI = 150                    # 输出图像DPI
FIGURE_SIZE = (18, 6)               # 对比图大小 (宽, 高) - 1行3列

# 5. 高级配置
UPSAMPLE_MODE = 'bicubic'           # 上采样模式: nearest, bilinear, bicubic
NUM_WORKERS = 0                     # 数据加载线程数 (0为主线程)

# ==================== END CONFIG ====================


# === 常量配置 ===
REPO_DIR = '../../dinov3'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AttentionExtractor:
    """
    注意力提取器 - 使用Hook机制提取DINOv3最后一层Transformer Block的注意力矩阵

    DINOv3使用scaled_dot_product_attention，默认不返回注意力权重。
    因此需要通过Hook进入attention模块，在Q、K计算后手动计算注意力矩阵。

    Token顺序: [CLS (idx 0), storage_tokens (idx 1..n), patch_tokens (idx n+1..end)]
    """

    def __init__(self, model: nn.Module):
        """
        初始化注意力提取器

        Args:
            model: DINOv3 Vision Transformer模型
        """
        self.model = model
        self.attention_map: Optional[torch.Tensor] = None
        self.hook_handle = None

        # Get storage tokens count from the underlying model
        # Handle both regular models and Peft-wrapped models
        if hasattr(model, 'model'):
            underlying = model.model  # PeftModel wraps the base model
        else:
            underlying = model

        self.n_storage_tokens = getattr(underlying, 'n_storage_tokens', 0)
        if self.n_storage_tokens > 0:
            print(f"  Detected {self.n_storage_tokens} register/storage tokens in model")

    def register_hook(self):
        """
        在模型最后一层Transformer Block的Attention模块上注册Forward Hook

        Hook函数会在attention forward过程中拦截，手动计算并保存注意力矩阵
        """
        # 定位到最后一层的SelfAttentionBlock
        # DINOv3结构: model.blocks = ModuleList([SelfAttentionBlock, ...])
        # Handle Peft-wrapped models
        if hasattr(self.model, 'model'):
            target_model = self.model.model
        else:
            target_model = self.model

        if not hasattr(target_model, 'blocks'):
            raise AttributeError("Model does not have 'blocks' attribute. Check model structure.")

        last_block = target_model.blocks[-1]  # 最后一个Transformer Block
        attention_module = last_block.attn   # SelfAttention模块

        # 定义Hook函数 - 在attention forward时自动调用
        def hook_fn(module, input, output):
            """
            Hook函数: 在attention计算过程中提取Q、K并计算注意力矩阵

            Args:
                module: 被hook的attention模块
                input: 输入元组 (x, attn_bias, rope)
                output: attention模块的输出 (未使用，我们自行计算)
            """
            # 从input中提取输入特征x
            x = input[0]  # [B, N, C] - B: batch, N: num_tokens, C: embed_dim
            rope = input[2] if len(input) > 2 else None

            # 获取模块参数
            num_heads = module.num_heads
            scale = module.scale  # head_dim ** -0.5

            # 1. 计算QKV - 与SelfAttention.compute_attention一致
            qkv = module.qkv(x)  # [B, N, 3*C]
            B, N, _ = qkv.shape
            C = module.qkv.in_features

            # 2. 重塑为多头格式 [B, N, 3, num_heads, head_dim]
            qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads)
            q, k, v = torch.unbind(qkv, 2)  # 每个: [B, N, num_heads, head_dim]

            # 3. 转置为 [B, num_heads, N, head_dim]
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

            # 4. 应用RoPE位置编码 (如果存在)
            if rope is not None:
                q, k = module.apply_rope(q, k, rope)

            # 5. 手动计算注意力矩阵 Attention = softmax(Q @ K^T / sqrt(d_k))
            # Q @ K^T: [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, N] -> [B, num_heads, N, N]
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)  # 在最后一个维度(N)上做softmax

            # 6. 保存注意力矩阵供后续使用
            # 形状: [Batch, Num_Heads, Num_Tokens, Num_Tokens]
            self.attention_map = attn_weights.detach().clone()

        # 注册forward hook到attention模块
        self.hook_handle = attention_module.register_forward_hook(hook_fn)

    def remove_hook(self):
        """移除注册的hook，释放资源"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def extract_cls_attention(self, expected_num_patches: int) -> Optional[torch.Tensor]:
        """
        从保存的注意力矩阵中提取[CLS] token对所有patch tokens的注意力

        DINOv3 Token顺序:
            - Index 0: CLS token
            - Index 1 to n_storage_tokens: storage/register tokens
            - Index n_storage_tokens+1 to end: patch tokens

        Args:
            expected_num_patches: 期望的patch token数量 (用于验证)

        Returns:
            注意力权重张量，形状: [Batch, Num_Heads, Num_PatchTokens]
            如果attention_map为None，返回None
        """
        if self.attention_map is None:
            return None

        # attention_map形状: [B, num_heads, N, N]
        # N = 1 (CLS) + n_storage_tokens + num_patches
        # [CLS] token位于索引0

        B, num_heads, N, _ = self.attention_map.shape

        # 提取CLS token (索引0) 对所有其他tokens的注意力
        # Shape: [B, num_heads, N-1]
        cls_to_all = self.attention_map[:, :, 0, 1:]

        # 跳过storage tokens，只保留patch tokens
        # storage tokens位于索引1..n_storage_tokens (在cls_to_all中是0..n_storage_tokens-1)
        # patch tokens从索引n_storage_tokens+1开始 (在cls_to_all中从n_storage_tokens开始)
        if self.n_storage_tokens > 0:
            cls_attention = cls_to_all[:, :, self.n_storage_tokens:]  # [B, num_heads, num_patches]
        else:
            cls_attention = cls_to_all

        # 验证patch数量
        actual_patches = cls_attention.shape[-1]
        if actual_patches != expected_num_patches:
            print(f"  Warning: Extracted patches ({actual_patches}) != expected ({expected_num_patches})")
            print(f"  This may indicate incorrect n_storage_tokens detection")

        return cls_attention

    def clear(self):
        """清除保存的注意力图，准备下一次提取"""
        self.attention_map = None


def load_models(device):
    """
    加载基础DINOv3模型和LoRA微调模型

    Args:
        device: 计算设备 (cuda/cpu)

    Returns:
        tuple: (base_model, finetuned_model)
    """
    print(f"Loading base DINOv3 model ({MODEL_NAME})...")
    base_model = torch.hub.load(
        REPO_DIR,
        MODEL_NAME,
        source='local',
        weights=BASE_WEIGHTS_PATH
    )

    # 构建完整的LoRA权重路径
    lora_weights_path = os.path.join(LORA_WEIGHTS_DIR, f'student_encoder_epoch_{LORA_EPOCH}')

    print(f"Loading LoRA adapters from {lora_weights_path}...")
    try:
        finetuned_model = PeftModel.from_pretrained(base_model, lora_weights_path)
        print("Successfully loaded Student model with LoRA!")
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        print("Falling back to base model (Verify your lora_path!)")
        finetuned_model = base_model

    finetuned_model.to(device)
    finetuned_model.eval()

    # 加载独立的原始模型用于对比
    print("Loading separate base model for comparison...")
    base_model_copy = torch.hub.load(
        REPO_DIR,
        MODEL_NAME,
        source='local',
        weights=BASE_WEIGHTS_PATH
    )
    base_model_copy.to(device)
    base_model_copy.eval()

    return base_model_copy, finetuned_model


def get_transforms():
    """
    获取图像预处理变换

    Returns:
        tuple: (transform_tensor, transform_resize)
    """
    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    transform_resize = transforms.RandomResizedCrop(
        size=TARGET_SIZE,
        scale=RESIZE_SCALE,
        ratio=RESIZE_RATIO,
        interpolation=transforms.InterpolationMode.BILINEAR
    )

    return transform_tensor, transform_resize


def preprocess_image(img_path: str, transform_resize, transform_tensor, device) -> Tuple[Image.Image, torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    预处理图像

    Args:
        img_path: 图像文件路径
        transform_resize: 随机裁剪变换
        transform_tensor: 张量变换
        device: 计算设备

    Returns:
        tuple: (原始图像PIL, 处理后的张量, 原始尺寸(H,W), 网格尺寸(h_grid,w_grid))
    """
    img = Image.open(img_path).convert('RGB')
    img = transform_resize(img)
    W_orig, H_orig = img.size

    # 调整尺寸为patch size的整数倍
    new_w = (W_orig * SCALE_FACTOR // PATCH_SIZE) * PATCH_SIZE
    new_h = (H_orig * SCALE_FACTOR // PATCH_SIZE) * PATCH_SIZE
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    img_tensor = transform_tensor(img_resized).unsqueeze(0).to(device)

    # 计算网格尺寸
    h_grid = new_h // PATCH_SIZE
    w_grid = new_w // PATCH_SIZE

    return img, img_tensor, (H_orig, W_orig), (h_grid, w_grid)


def extract_attention_with_model(model: nn.Module, img_tensor: torch.Tensor, extractor: AttentionExtractor,
                                  expected_h_grid: int, expected_w_grid: int) -> np.ndarray:
    """
    使用AttentionExtractor提取[CLS] token的注意力图

    Args:
        model: DINOv3模型
        img_tensor: 输入图像张量 [1, 3, H, W]
        extractor: 已注册hook的AttentionExtractor实例
        expected_h_grid: 期望的高度方向patch数量
        expected_w_grid: 期望的宽度方向patch数量

    Returns:
        numpy数组: 平均注意力图，形状 [h_grid, w_grid]
    """
    with torch.no_grad():
        # 清空之前的注意力图
        extractor.clear()

        # 前向传播 - hook会自动捕获注意力
        _ = model.forward_features(img_tensor)

        # 计算期望的patch数量
        expected_num_patches = expected_h_grid * expected_w_grid

        # 提取CLS注意力
        cls_attn = extractor.extract_cls_attention(expected_num_patches)  # [1, num_heads, num_patches]

        if cls_attn is None:
            raise RuntimeError("Failed to extract attention. Hook may not be working properly.")

        # 将所有head的注意力求平均 -> [num_patches]
        cls_attn_mean = cls_attn.mean(dim=1).squeeze(0)  # [num_patches]

        # 验证数量
        if cls_attn_mean.shape[0] != expected_num_patches:
            print(f"  Warning: Got {cls_attn_mean.shape[0]} patches, expected {expected_num_patches}")
            print(f"  Auto-adjusting grid size...")
            # Try to find the best fit
            n = cls_attn_mean.shape[0]
            # Find closest factorization
            h_grid = int(np.sqrt(n))
            while n % h_grid != 0:
                h_grid -= 1
            w_grid = n // h_grid
            print(f"  New grid: {h_grid}x{w_grid}")
        else:
            h_grid, w_grid = expected_h_grid, expected_w_grid

        # reshape为2D空间图 [h_grid, w_grid]
        attention_map = cls_attn_mean.cpu().numpy().reshape(h_grid, w_grid)

        return attention_map


def upsample_attention(attention_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    将注意力图上采样到目标尺寸

    Args:
        attention_map: 低分辨率注意力图 [h_grid, w_grid]
        target_size: 目标尺寸 (H, W)

    Returns:
        上采样后的注意力图 [H, W]
    """
    # 转换为torch张量进行上采样 [1, 1, h, w]
    attn_tensor = torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float()

    mode_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic'
    }
    mode = mode_map.get(UPSAMPLE_MODE, 'bicubic')
    align_corners = None if mode == 'nearest' else False

    upsampled = F.interpolate(
        attn_tensor,
        size=target_size,
        mode=mode,
        align_corners=align_corners
    )

    # 转回numpy并squeeze
    result = upsampled.squeeze().numpy()

    return result


def normalize_attention(attention_map: np.ndarray) -> np.ndarray:
    """
    归一化注意力图到[0, 1]范围

    Args:
        attention_map: 原始注意力图

    Returns:
        归一化后的注意力图
    """
    attn_min = attention_map.min()
    attn_max = attention_map.max()

    if attn_max > attn_min:
        return (attention_map - attn_min) / (attn_max - attn_min)
    else:
        return np.zeros_like(attention_map)


def create_overlay_image(gray_img: np.ndarray, attention_map: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """
    创建注意力热力图与原图的叠加效果

    Args:
        gray_img: 灰度原图 [H, W, 3] 或 [H, W]
        attention_map: 注意力热力图 [H, W]，值范围[0, 1]
        alpha: 叠加透明度

    Returns:
        叠加后的RGB图像 [H, W, 3]
    """
    # 确保灰度图是3通道
    if len(gray_img.shape) == 2:
        gray_img = np.stack([gray_img] * 3, axis=-1)

    # 使用matplotlib colormap将注意力图转换为RGB
    cmap = plt.get_cmap(COLORMAP_ATTENTION)
    heatmap_rgb = cmap(attention_map)[:, :, :3]  # 去除alpha通道

    # 混合: result = (1-alpha) * gray_img + alpha * heatmap
    overlay = (1 - alpha) * gray_img + alpha * heatmap_rgb

    # 裁剪到有效范围
    overlay = np.clip(overlay, 0, 1)

    return overlay


def save_attention_comparison(
    original_img: Image.Image,
    attention_base: np.ndarray,
    attention_finetuned: np.ndarray,
    out_path: str
):
    """
    保存1x3对比图: [原图灰度] | [Base模型注意力] | [Finetuned模型注意力]

    Args:
        original_img: 原始PIL图像
        attention_base: Base模型注意力图
        attention_finetuned: Finetuned模型注意力图
        out_path: 输出文件路径
    """
    # 转换原图为numpy数组并归一化到[0, 1]
    img_array = np.array(original_img).astype(np.float32) / 255.0

    # 转换为灰度图用于显示
    if len(img_array.shape) == 3:
        gray_img = np.mean(img_array, axis=2)  # [H, W]
    else:
        gray_img = img_array

    # 归一化gray_img到[0, 1]
    gray_img = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min() + 1e-8)
    gray_img_rgb = np.stack([gray_img] * 3, axis=-1)

    # 归一化注意力图
    attn_base_norm = normalize_attention(attention_base)
    attn_finetuned_norm = normalize_attention(attention_finetuned)

    # 创建叠加图
    overlay_base = create_overlay_image(gray_img_rgb, attn_base_norm)
    overlay_finetuned = create_overlay_image(gray_img_rgb, attn_finetuned_norm)

    # 创建1x3子图
    fig, axs = plt.subplots(1, 3, figsize=FIGURE_SIZE)

    # 1. 原始灰度图
    axs[0].imshow(gray_img, cmap='gray')
    axs[0].set_title("Original Image (Grayscale)", fontsize=12)
    axs[0].axis('off')

    # 2. Base模型注意力叠加
    axs[1].imshow(overlay_base)
    axs[1].set_title("Base Model Attention", fontsize=12)
    axs[1].axis('off')

    # 3. Finetuned模型注意力叠加
    axs[2].imshow(overlay_finetuned)
    axs[2].set_title(f"Finetuned Model Attention (Epoch {LORA_EPOCH})", fontsize=12)
    axs[2].axis('off')

    plt.tight_layout()
    out_path_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path_abs), exist_ok=True)
    plt.savefig(out_path_abs, bbox_inches='tight', dpi=OUTPUT_DPI)
    plt.close()

    print(f"  Saved attention comparison to: {out_path_abs}")


def save_detailed_comparison(
    original_img: Image.Image,
    attention_base: np.ndarray,
    attention_finetuned: np.ndarray,
    out_path: str
):
    """
    保存详细的2x3对比图:
    第一行: [原图] | [Base纯热力图] | [Base叠加图]
    第二行: [原图] | [Finetuned纯热力图] | [Finetuned叠加图]

    Args:
        original_img: 原始PIL图像
        attention_base: Base模型注意力图
        attention_finetuned: Finetuned模型注意力图
        out_path: 输出文件路径
    """
    # 转换原图为numpy数组
    img_array = np.array(original_img).astype(np.float32) / 255.0

    # 转换为灰度图
    if len(img_array.shape) == 3:
        gray_img = np.mean(img_array, axis=2)
    else:
        gray_img = img_array

    gray_img = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min() + 1e-8)
    gray_img_rgb = np.stack([gray_img] * 3, axis=-1)

    # 归一化注意力图
    attn_base_norm = normalize_attention(attention_base)
    attn_finetuned_norm = normalize_attention(attention_finetuned)

    # 创建叠加图
    overlay_base = create_overlay_image(gray_img_rgb, attn_base_norm)
    overlay_finetuned = create_overlay_image(gray_img_rgb, attn_finetuned_norm)

    # 创建2x3子图
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # 第一行 - Base Model
    axs[0, 0].imshow(gray_img, cmap='gray')
    axs[0, 0].set_title("Original Image", fontsize=12)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(attn_base_norm, cmap=COLORMAP_ATTENTION)
    axs[0, 1].set_title("Base Model - Attention Heatmap", fontsize=12)
    axs[0, 1].axis('off')

    axs[0, 2].imshow(overlay_base)
    axs[0, 2].set_title("Base Model - Overlay", fontsize=12)
    axs[0, 2].axis('off')

    # 第二行 - Finetuned Model
    axs[1, 0].imshow(gray_img, cmap='gray')
    axs[1, 0].set_title("Original Image", fontsize=12)
    axs[1, 0].axis('off')

    axs[1, 1].imshow(attn_finetuned_norm, cmap=COLORMAP_ATTENTION)
    axs[1, 1].set_title(f"Finetuned (Epoch {LORA_EPOCH}) - Attention Heatmap", fontsize=12)
    axs[1, 1].axis('off')

    axs[1, 2].imshow(overlay_finetuned)
    axs[1, 2].set_title(f"Finetuned (Epoch {LORA_EPOCH}) - Overlay", fontsize=12)
    axs[1, 2].axis('off')

    plt.tight_layout()
    out_path_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path_abs), exist_ok=True)
    plt.savefig(out_path_abs, bbox_inches='tight', dpi=OUTPUT_DPI)
    plt.close()

    print(f"  Saved detailed comparison to: {out_path_abs}")


def process_single_image(
    img_path: str,
    out_folder: str,
    base_model: nn.Module,
    finetuned_model: nn.Module,
    extractor_base: AttentionExtractor,
    extractor_finetuned: AttentionExtractor,
    transform_tensor,
    transform_resize,
    device
):
    """
    处理单张图像，提取并可视化注意力

    Args:
        img_path: 图像路径
        out_folder: 输出文件夹
        base_model: Base DINOv3模型
        finetuned_model: LoRA微调后的模型
        extractor_base: Base模型的注意力提取器
        extractor_finetuned: Finetuned模型的注意力提取器
        transform_tensor: 张量变换
        transform_resize: 尺寸调整变换
        device: 计算设备
    """
    os.makedirs(out_folder, exist_ok=True)

    # 预处理图像
    img, img_tensor, (H_orig, W_orig), (h_grid, w_grid) = preprocess_image(
        img_path, transform_resize, transform_tensor, device
    )

    print(f"  Image size: {W_orig}x{H_orig}, Patch grid: {w_grid}x{h_grid}")

    # 提取Base模型的注意力
    print(f"  Extracting attention from base model...")
    attention_base = extract_attention_with_model(base_model, img_tensor, extractor_base, h_grid, w_grid)

    # 提取Finetuned模型的注意力
    print(f"  Extracting attention from finetuned model...")
    attention_finetuned = extract_attention_with_model(finetuned_model, img_tensor, extractor_finetuned, h_grid, w_grid)

    # 上采样到原始尺寸
    attention_base_full = upsample_attention(attention_base, (H_orig, W_orig))
    attention_finetuned_full = upsample_attention(attention_finetuned, (H_orig, W_orig))

    # 构建输出路径
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)

    # 保存2x3详细对比图
    out_path_detailed = os.path.join(out_folder, f"{name}_attention_detailed{ext}")
    save_detailed_comparison(img, attention_base_full, attention_finetuned_full, out_path_detailed)


def main():
    """主函数"""
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # 加载模型
    base_model, finetuned_model = load_models(device)

    # 创建注意力提取器并注册hooks
    print("\nRegistering attention hooks...")
    extractor_base = AttentionExtractor(base_model)
    extractor_base.register_hook()
    print("  Base model hook registered on last block's attention")

    extractor_finetuned = AttentionExtractor(finetuned_model)
    extractor_finetuned.register_hook()
    print("  Finetuned model hook registered on last block's attention")

    # 获取图像变换
    transform_tensor, transform_resize = get_transforms()

    # 处理图像
    input_dir = Path(INPUT_IMAGE_DIR)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {INPUT_IMAGE_DIR}")
        return

    # 收集所有图像
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(input_dir.glob(ext))
    images = sorted(images)

    if not images:
        print(f"Warning: No images found in {INPUT_IMAGE_DIR}")
        print(f"Supported extensions: {IMAGE_EXTENSIONS}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nFound {len(images)} images to process")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Colormap: {COLORMAP_ATTENTION}")
    print("-" * 50)

    # 处理每张图像
    try:
        for img_path in images:
            print(f"Processing: {img_path.name}")
            process_single_image(
                str(img_path), OUTPUT_DIR, base_model, finetuned_model,
                extractor_base, extractor_finetuned,
                transform_tensor, transform_resize, device
            )
            print()
    finally:
        # 清理hooks
        print("Removing attention hooks...")
        extractor_base.remove_hook()
        extractor_finetuned.remove_hook()

    print("-" * 50)
    print(f"All processing complete! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
