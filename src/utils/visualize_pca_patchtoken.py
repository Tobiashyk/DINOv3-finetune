"""
PCA Visualization Script for DINOv3 with LoRA Fine-tuning

This script compares the base DINOv3 model and the LoRA-finetuned model
by visualizing patch features using PCA (1 and 3 components).

Usage:
    python visualize_pca.py

To modify settings, edit the CONFIG section below.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from pathlib import Path
import numpy as np
from peft import PeftModel




def load_models(device):
    """加载基础模型和LoRA微调模型"""
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
        model = PeftModel.from_pretrained(base_model, lora_weights_path)
        print("Successfully loaded Student model with LoRA!")
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        print("Falling back to base model (Verify your lora_path!)")
        model = base_model

    model.to(device)
    model.eval()

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

    return base_model_copy, model


def get_transforms():
    """获取图像预处理变换"""
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


def preprocess_image(img_path, transform_resize, transform_tensor, device):
    """预处理图像"""
    img = Image.open(img_path).convert('RGB')
    img = transform_resize(img)
    W_orig, H_orig = img.size

    new_w = (W_orig * SCALE_FACTOR // PATCH_SIZE) * PATCH_SIZE
    new_h = (H_orig * SCALE_FACTOR // PATCH_SIZE) * PATCH_SIZE
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    img_tensor = transform_tensor(img_resized).unsqueeze(0).to(device)

    h_grid = new_h // PATCH_SIZE
    w_grid = new_w // PATCH_SIZE

    return img, img_tensor, (H_orig, W_orig), (h_grid, w_grid)


def extract_features(model, img_tensor):
    """提取模型特征"""
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        patches = features["x_norm_patchtokens"].squeeze(0).cpu().numpy()
    return patches


def compute_pca(patches, n_components, h_grid, w_grid, use_two_stage=None, energy_percentile=None):
    """
    计算PCA并生成热力图。

    当 n_components == 1 时：使用标准单次PCA。
    当 n_components == 3 时：可选择使用两阶段PCA或标准PCA：
        - 两阶段PCA: 基于能量阈值分离前景/背景，仅对前景做3D PCA
        - 标准PCA: 对所有patch做3D PCA（适用于本身已学好分离的模型）

    Args:
        patches: [n_patches, feature_dim] 特征数组
        n_components: PCA组件数 (1 或 3)
        h_grid, w_grid: 网格尺寸
        use_two_stage: 是否使用两阶段PCA。None=使用全局配置USE_TWO_STAGE_PCA
        energy_percentile: 能量阈值百分位。None=使用全局配置(35%或分别配置)
    """
    if use_two_stage is None:
        use_two_stage = USE_TWO_STAGE_PCA
    if energy_percentile is None:
        energy_percentile = BASE_ENERGY_PERCENTILE  # 默认使用Base模型的阈值

    n_patches = patches.shape[0]

    if n_components == 1:
        # ===== 标准单次PCA (保持不变) =====
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(patches)
        heatmap = pca_result.reshape(h_grid, w_grid)

        # 百分位数裁剪
        p_low, p_high = PERCENTILE_CLIP
        heatmap = np.clip(heatmap, np.percentile(heatmap, p_low), np.percentile(heatmap, p_high))

        # 归一化到 [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    elif n_components == 3:
        # 如果禁用两阶段PCA，直接使用标准3D PCA
        if not use_two_stage:
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(patches)
            heatmap = pca_result.reshape(h_grid, w_grid, 3)
            p_low, p_high = PERCENTILE_CLIP
            heatmap = np.clip(heatmap, np.percentile(heatmap, p_low), np.percentile(heatmap, p_high))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap

        # ===== 两阶段PCA (Two-stage PCA) =====

        # Stage 1: 背景剔除 - 基于Patch能量(L2范数)提取前景掩码
        # 原理：原子区域特征响应强（能量高），背景区域能量低
        patch_energy = np.linalg.norm(patches, axis=1)  # [n_patches]

        # 使用能量分位数来分离前景/背景
        # 使用指定的阈值，保留更多高能量区域（原子和边缘）
        target_percentile = energy_percentile  # 越小保留越多patch，越大剔除越多背景
        energy_threshold = np.percentile(patch_energy, target_percentile)
        foreground_mask = patch_energy > energy_threshold
        foreground_indices = np.where(foreground_mask)[0]

        print(f"  Foreground: high energy patches ({len(foreground_indices)}/{n_patches}, "
              f"threshold={energy_threshold:.3f}, percentile={target_percentile}%)")

        # 自适应调整：确保前景比例在合理范围内(20%-60%)
        if len(foreground_indices) > n_patches * 0.6:  # 前景太多，提高阈值
            target_percentile = 50
            energy_threshold = np.percentile(patch_energy, target_percentile)
            foreground_mask = patch_energy > energy_threshold
            foreground_indices = np.where(foreground_mask)[0]
            print(f"  Adjusted: too many patches, new threshold ({len(foreground_indices)}/{n_patches}, "
                  f"percentile={target_percentile}%)")
        elif len(foreground_indices) < n_patches * 0.2:  # 前景太少，降低阈值
            target_percentile = 20
            energy_threshold = np.percentile(patch_energy, target_percentile)
            foreground_mask = patch_energy > energy_threshold
            foreground_indices = np.where(foreground_mask)[0]
            print(f"  Adjusted: too few patches, new threshold ({len(foreground_indices)}/{n_patches}, "
                  f"percentile={target_percentile}%)")

        # 如果前景仍然太少，退化为标准3D PCA
        if len(foreground_indices) < 10:
            print(f"  Warning: Only {len(foreground_indices)} foreground patches found. "
                  "Falling back to standard 3D PCA.")
            pca_3d = PCA(n_components=3)
            pca_3d_result = pca_3d.fit_transform(patches)
            heatmap = pca_3d_result.reshape(h_grid, w_grid, 3)
            p_low, p_high = PERCENTILE_CLIP
            heatmap = np.clip(heatmap, np.percentile(heatmap, p_low), np.percentile(heatmap, p_high))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap

        # Stage 2: 仅对前景patch进行3维PCA
        foreground_patches = patches[foreground_mask]  # [n_foreground, feature_dim]
        pca_3d = PCA(n_components=3)
        foreground_3d = pca_3d.fit_transform(foreground_patches)  # [n_foreground, 3]

        # Stage 3: 空间重构与归一化
        # 创建全黑的背景数组 [n_patches, 3]
        heatmap_flat = np.zeros((n_patches, 3), dtype=np.float32)

        # 将3维前景特征填入对应位置
        heatmap_flat[foreground_indices] = foreground_3d

        # 仅对前景数值进行百分位数裁剪和Min-Max归一化
        if len(foreground_indices) > 0:
            foreground_values = heatmap_flat[foreground_indices]  # [n_foreground, 3]

            # 百分位数裁剪（仅针对前景）
            p_low, p_high = PERCENTILE_CLIP
            v_low = np.percentile(foreground_values, p_low)
            v_high = np.percentile(foreground_values, p_high)
            foreground_values = np.clip(foreground_values, v_low, v_high)

            # Min-Max归一化到 [0, 1]（仅针对前景）
            f_min, f_max = foreground_values.min(), foreground_values.max()
            if f_max > f_min:
                foreground_values = (foreground_values - f_min) / (f_max - f_min)
            else:
                foreground_values = np.zeros_like(foreground_values)

            # 将归一化后的前景值填回
            heatmap_flat[foreground_indices] = foreground_values

        # 保持背景为严格的 [0, 0, 0]（黑色）
        # reshape为热力图形状
        heatmap = heatmap_flat.reshape(h_grid, w_grid, 3)

        return heatmap

    else:
        # 其他组件数使用标准PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(patches)
        heatmap = pca_result.reshape(h_grid, w_grid, n_components)
        p_low, p_high = PERCENTILE_CLIP
        heatmap = np.clip(heatmap, np.percentile(heatmap, p_low), np.percentile(heatmap, p_high))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap


def upsample_heatmap(heatmap, target_size, is_rgb=False):
    """上采样热力图到目标尺寸"""
    if is_rgb or len(heatmap.shape) == 3:
        # RGB 图像 [H, W, C] -> [1, C, H, W]
        heatmap_tensor = torch.from_numpy(heatmap).permute(2, 0, 1).unsqueeze(0).float()
    else:
        # 灰度图像 [H, W] -> [1, 1, H, W]
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()

    mode_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic'
    }
    mode = mode_map.get(UPSAMPLE_MODE, 'bicubic')
    align_corners = None if mode == 'nearest' else False

    upsampled = torch.nn.functional.interpolate(
        heatmap_tensor,
        size=target_size,
        mode=mode,
        align_corners=align_corners
    )

    result = upsampled.squeeze().numpy()
    if is_rgb or len(heatmap.shape) == 3:
        result = result.transpose(1, 2, 0) if len(result.shape) == 3 else result

    return np.clip(result, 0, 1)


def save_comparison_figure(img, heatmap_base, heatmap_finetuned, out_path, n_components):
    """保存对比图"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=FIGURE_SIZE)

    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    if n_components == 1:
        axs[1].imshow(heatmap_base, cmap=COLORMAP_1C)
        axs[2].imshow(heatmap_finetuned, cmap=COLORMAP_1C)
    else:
        axs[1].imshow(heatmap_base)
        axs[2].imshow(heatmap_finetuned)

    axs[1].set_title(f"Base Model PCA ({n_components}C)")
    axs[1].axis('off')
    axs[2].set_title(f"Finetuned Model PCA ({n_components}C)")
    axs[2].axis('off')

    plt.savefig(out_path, bbox_inches='tight', dpi=OUTPUT_DPI)
    plt.close()


def save_combined_figure(img, heatmap_base_1c, heatmap_finetuned_1c,
                         heatmap_base_3c, heatmap_finetuned_3c, out_path):
    """保存组合对比图 (2行3列: 第一行PCA1, 第二行PCA3)"""
    fig, axs = plt.subplots(2, 3, figsize=FIGURE_SIZE)

    # 第一行: 原图 / Base PCA1 / Finetuned PCA1
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(heatmap_base_1c, cmap=COLORMAP_1C)
    axs[0, 1].set_title("Base Model PCA (1C)")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(heatmap_finetuned_1c, cmap=COLORMAP_1C)
    axs[0, 2].set_title("Finetuned Model PCA (1C)")
    axs[0, 2].axis('off')

    # 第二行: 原图 / Base PCA3 / Finetuned PCA3
    axs[1, 0].imshow(img)
    axs[1, 0].set_title("Original Image")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(heatmap_base_3c)
    axs[1, 1].set_title("Base Model PCA (3C)")
    axs[1, 1].axis('off')

    axs[1, 2].imshow(heatmap_finetuned_3c)
    axs[1, 2].set_title("Finetuned Model PCA (3C)")
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=OUTPUT_DPI)
    plt.close()


def process_single_image(img_path, out_folder, base_model, finetuned_model,
                         transform_tensor, transform_resize, device, n_components):
    """处理单张图像"""
    os.makedirs(out_folder, exist_ok=True)

    img, img_tensor, (H_orig, W_orig), (h_grid, w_grid) = preprocess_image(
        img_path, transform_resize, transform_tensor, device
    )

    # 提取特征
    base_patches = extract_features(base_model, img_tensor)
    finetuned_patches = extract_features(finetuned_model, img_tensor)

    # 计算PCA
    # Base模型: 使用较高阈值(35%)剔除强背景噪声
    # Finetuned模型: 使用较低阈值(15%)保留更多原子，同时剔除低能量背景
    heatmap_base = compute_pca(base_patches, n_components, h_grid, w_grid,
                               use_two_stage=True, energy_percentile=BASE_ENERGY_PERCENTILE)
    heatmap_finetuned = compute_pca(finetuned_patches, n_components, h_grid, w_grid,
                                    use_two_stage=True, energy_percentile=FINETUNED_ENERGY_PERCENTILE)

    # 上采样
    is_rgb = n_components == 3
    heatmap_base_full = upsample_heatmap(heatmap_base, (H_orig, W_orig), is_rgb)
    heatmap_finetuned_full = upsample_heatmap(heatmap_finetuned, (H_orig, W_orig), is_rgb)

    # 构建输出路径
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    out_name = f"{name}_pca{n_components}C{ext}"
    out_path = os.path.join(out_folder, out_name)

    # 保存对比图
    save_comparison_figure(img, heatmap_base_full, heatmap_finetuned_full, out_path, n_components)

    print(f"  PCA-{n_components}C saved to: {out_name}")


def process_image_all_components(img_path, out_folder, base_model, finetuned_model,
                                  transform_tensor, transform_resize, device):
    """处理单张图像，同时生成PCA1和PCA3的组合图 (2行3列)"""
    os.makedirs(out_folder, exist_ok=True)

    img, img_tensor, (H_orig, W_orig), (h_grid, w_grid) = preprocess_image(
        img_path, transform_resize, transform_tensor, device
    )

    # 提取特征
    base_patches = extract_features(base_model, img_tensor)
    finetuned_patches = extract_features(finetuned_model, img_tensor)

    # 计算PCA1 (标准PCA)
    heatmap_base_1c = compute_pca(base_patches, 1, h_grid, w_grid,
                                   use_two_stage=False, energy_percentile=BASE_ENERGY_PERCENTILE)
    heatmap_finetuned_1c = compute_pca(finetuned_patches, 1, h_grid, w_grid,
                                        use_two_stage=False, energy_percentile=FINETUNED_ENERGY_PERCENTILE)

    # 计算PCA3 (标准PCA，逻辑与PCA1相同，仅组件数=3)
    heatmap_base_3c = compute_pca(base_patches, 3, h_grid, w_grid,
                                   use_two_stage=False, energy_percentile=BASE_ENERGY_PERCENTILE)
    heatmap_finetuned_3c = compute_pca(finetuned_patches, 3, h_grid, w_grid,
                                        use_two_stage=False, energy_percentile=FINETUNED_ENERGY_PERCENTILE)

    # 上采样
    heatmap_base_1c_full = upsample_heatmap(heatmap_base_1c, (H_orig, W_orig), is_rgb=False)
    heatmap_finetuned_1c_full = upsample_heatmap(heatmap_finetuned_1c, (H_orig, W_orig), is_rgb=False)
    heatmap_base_3c_full = upsample_heatmap(heatmap_base_3c, (H_orig, W_orig), is_rgb=True)
    heatmap_finetuned_3c_full = upsample_heatmap(heatmap_finetuned_3c, (H_orig, W_orig), is_rgb=True)

    # 构建输出路径
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    out_name = f"{name}_combined_pca{ext}"
    out_path = os.path.join(out_folder, out_name)

    # 保存组合图 (2行3列)
    save_combined_figure(img, heatmap_base_1c_full, heatmap_finetuned_1c_full,
                         heatmap_base_3c_full, heatmap_finetuned_3c_full, out_path)

    print(f"  Combined PCA saved to: {out_name}")


def main():
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # 加载模型
    base_model, finetuned_model = load_models(device)

    # 获取变换
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

    print(f"\nFound {len(images)} images to process")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"PCA components: {PCA_COMPONENTS_LIST}")
    print(f"Scale factor: {SCALE_FACTOR}")
    print("-" * 50)

    # 处理每张图像
    for img_path in images:
        print(f"Processing: {img_path.name}")
        process_image_all_components(
            str(img_path), OUTPUT_DIR, base_model, finetuned_model,
            transform_tensor, transform_resize, device
        )

    print("-" * 50)
    print(f"All processing complete! Results saved to: {OUTPUT_DIR}")


# ==================== CONFIG ====================
# 修改以下配置参数以适应你的需求
for i in range(2,3):
    # 1. 模型路径配置
    BASE_WEIGHTS_PATH = '../../weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
    LORA_WEIGHTS = 'old_dinoloss/trainall_C_and_MoS2_512_batch32_noisyall'  # 要改的
    LORA_WEIGHTS_DIR = '../../student_weights/' + LORA_WEIGHTS
    LORA_EPOCH = str(i*10)  # 要改的，确保与训练时保存的epoch一致
    MODEL_NAME = 'dinov3_vits16plus'  # 可选: dinov3_vits16plus, dinov3_vits16, dinov3_vitb16, etc.
    size = 2048

    # 2. 输入输出路径配置
    INPUT_IMAGE = 'stem_wash_test'  # 要改的
    # INPUT_IMAGE = 'MoS2_2H_test_only5'  # 要改的
    INPUT_IMAGE_DIR = '../../data/test_pic/' + INPUT_IMAGE
    # INPUT_IMAGE_DIR = '../../data/koala_test/'
    # OUTPUT_DIR = '../../data/visualize_pca/' + 'koala' + '/' + LORA_WEIGHTS + '/' + 'weights_epoch' + LORA_EPOCH + '/' + 'size' + str(size)
    OUTPUT_DIR = '../../data/visualize_pca/' + INPUT_IMAGE + '/' + LORA_WEIGHTS + '/' + 'weights_epoch' + LORA_EPOCH + '/' + 'size' + str(size)
    IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp']  # 支持的图像格式

    # 3. 图像处理配置
    SCALE_FACTOR = 1                    # 图像放大倍数 (1=原始大小, 2=放大2倍, etc.)
    PATCH_SIZE = 16                     # 模型 patch size (通常为16)
    TARGET_SIZE = (size, size)          # 图像预处理目标尺寸 (H, W)
    RESIZE_SCALE = (1.0, 1.0)           # RandomResizedCrop scale
    RESIZE_RATIO = (1.0, 1.0)           # RandomResizedCrop aspect ratio

    # 4. PCA 可视化配置
    PCA_COMPONENTS_LIST = [1, 3]        # 要生成的PCA组件数 [1, 3] 表示生成1组件和3组件图
    COLORMAP_1C = 'inferno'             # 1组件PCA使用的颜色映射
    OUTPUT_DPI = 150                    # 输出图像DPI
    FIGURE_SIZE = (18, 12)              # 六图对比的图像大小 (宽, 高) - 2行3列

    # 5. 高级配置
    PERCENTILE_CLIP = [5, 95]           # 用于热力图裁剪的百分位数 [low, high]
    UPSAMPLE_MODE = 'bicubic'           # 上采样模式: nearest, bilinear, bicubic
    NUM_WORKERS = 0                     # 数据加载线程数 (0为主线程)

    # 6. 两阶段PCA配置
    USE_TWO_STAGE_PCA = True            # 是否启用两阶段PCA (False=对所有patch做标准3D PCA)

    # 为Base和Finetuned模型分别配置能量阈值
    # Base模型: 需要较高阈值(35%)来剔除强背景噪声
    # Finetuned模型: 使用较低阈值(15%)，保留更多原子同时剔除低能量背景
    BASE_ENERGY_PERCENTILE = 35         # Base模型能量阈值百分位
    FINETUNED_ENERGY_PERCENTILE = 15    # Finetuned模型能量阈值百分位 (更小=保留更多)

    ADAPTIVE_FOREGROUND_RANGE = (0.2, 0.6)  # 前景比例目标范围 (min, max)

    # ==================== END CONFIG ====================


    # === 常量配置 ===
    REPO_DIR = '../../dinov3'
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    if __name__ == "__main__":
        main()
