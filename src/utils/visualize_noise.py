import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from pathlib import Path
import argparse
import numpy as np
from peft import PeftModel

# === 设置路径 ===
REPO_DIR = '../../dinov3'
BASE_WEIGHTS_PATH = '../../weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 图像预处理
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_resize = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR)


def extract_features(model, img_tensor):
    """
    提取模型特征（直接使用encoder输出的patch tokens）
    Args:
        model: DINOv3模型（带有LoRA）
        img_tensor: 输入图像tensor
    Returns:
        patch_features: 用于PCA的patch特征 [N, D]
    """
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        patch_tokens = features["x_norm_patchtokens"]  # [B, N, D]
        return patch_tokens.squeeze(0).cpu().numpy()  # [N, D]


def apply_pca(features, patch_h, patch_w, n_components=1):
    """
    对特征应用PCA并归一化
    Args:
        features: [N, D] 特征数组
        patch_h: patch网格高度
        patch_w: patch网格宽度
        n_components: PCA组件数
    Returns:
        heatmap: 归一化的热力图
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features)
    heatmap = pca_result.reshape(patch_h, patch_w, n_components)

    # 归一化到[0, 1]
    p5, p95 = np.percentile(heatmap, [5, 95])
    heatmap = np.clip(heatmap, p5, p95)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap


def upsample_heatmap(heatmap, target_h, target_w):
    """
    上采样热力图到目标尺寸
    """
    if heatmap.ndim == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    else:
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).unsqueeze(0).float()

    upsampled = torch.nn.functional.interpolate(
        heatmap,
        size=(target_h, target_w),
        mode='bicubic',
        align_corners=False
    )

    result = upsampled.squeeze().numpy()
    if result.ndim == 2:
        return np.clip(result, 0, 1)
    else:
        return np.clip(result.transpose(1, 2, 0), 0, 1)


def visualize_comparison(img_path, out_folder,
                         stage1_model, stage2_model,
                         stage1_name="Stage 1", stage2_name="Stage 2",
                         n_components=1):
    """
    可视化对比：原图 + Stage 1 PCA + Stage 2 PCA
    """
    os.makedirs(out_folder, exist_ok=True)

    # 加载并预处理图像
    img = Image.open(img_path).convert('RGB')
    img_resized = transform_resize(img)
    H_orig, W_orig = img_resized.size[1], img_resized.size[0]

    # 确保尺寸是16的倍数（DINOv3使用16x16 patch）
    new_h = (H_orig // 16) * 16
    new_w = (W_orig // 16) * 16
    img_resized = img_resized.resize((new_w, new_h), resample=Image.BICUBIC)

    img_tensor = transform_tensor(img_resized).unsqueeze(0).to(device)

    # 计算patch网格尺寸
    h_grid = new_h // 16
    w_grid = new_w // 16

    print(f"Processing: {img_path.name} -> Grid: {h_grid}x{w_grid}")

    # === 提取两个阶段的特征（直接使用encoder输出）===
    stage1_features = extract_features(stage1_model, img_tensor)
    stage2_features = extract_features(stage2_model, img_tensor)

    # === PCA分析 ===
    heatmap_stage1 = apply_pca(stage1_features, h_grid, w_grid, n_components)
    heatmap_stage2 = apply_pca(stage2_features, h_grid, w_grid, n_components)

    # 上采样到原图尺寸
    heatmap_stage1_full = upsample_heatmap(heatmap_stage1, H_orig, W_orig)
    heatmap_stage2_full = upsample_heatmap(heatmap_stage2, H_orig, W_orig)

    # === 三图可视化 ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 原图
    axs[0].imshow(img_resized.resize((W_orig, H_orig)))
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Stage 1 PCA
    if n_components == 1:
        axs[1].imshow(heatmap_stage1_full, cmap='inferno')
    else:
        axs[1].imshow(heatmap_stage1_full)
    axs[1].set_title(f"{stage1_name} PCA")
    axs[1].axis('off')

    # Stage 2 PCA
    if n_components == 1:
        axs[2].imshow(heatmap_stage2_full, cmap='inferno')
    else:
        axs[2].imshow(heatmap_stage2_full)
    axs[2].set_title(f"{stage2_name} PCA")
    axs[2].axis('off')

    # 保存
    out_path = os.path.join(out_folder, os.path.basename(img_path))
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved: {out_path}")


def load_stage1_model(lora_weights_path):
    """加载Stage 1模型（只有LoRA）"""
    print(f"\n=== Loading Stage 1 Model ===")
    print(f"Loading base DINOv3...")
    base_model = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=BASE_WEIGHTS_PATH)

    print(f"Loading Stage 1 LoRA from: {lora_weights_path}")
    try:
        model = PeftModel.from_pretrained(base_model, lora_weights_path)
        print("Stage 1: Successfully loaded LoRA!")
    except Exception as e:
        print(f"Error loading Stage 1 LoRA: {e}")
        raise

    model.to(device)
    model.eval()

    return model


def load_stage2_model(lora_weights_path):
    """加载Stage 2模型（只加载LoRA）"""
    print(f"\n=== Loading Stage 2 Model ===")
    print(f"Loading base DINOv3...")
    base_model = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=BASE_WEIGHTS_PATH)

    print(f"Loading Stage 2 LoRA from: {lora_weights_path}")
    try:
        model = PeftModel.from_pretrained(base_model, lora_weights_path)
        print("Stage 2: Successfully loaded LoRA!")
    except Exception as e:
        print(f"Error loading Stage 2 LoRA: {e}")
        raise

    model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="PCA Comparison: Stage 1 vs Stage 2 Denoising")
    parser.add_argument("--stage1_path", type=str, default="../../student_weights/improved_clean_centering/student_encoder_epoch_40",
                        help="Stage 1 LoRA权重路径 (e.g., student_weights/.../student_encoder_epoch_40)")
    parser.add_argument("--stage2_path", type=str, default="../../student_weights/student_weights/stage2_noise/stage2_epoch_40/encoder",
                        help="Stage 2 LoRA权重路径 (e.g., student_weights/.../stage2_epoch_40/encoder)")
    parser.add_argument("--input_dir", type=str, default="../../data/test_pic/graphene_AA_256",
                        help="输入图像文件夹路径")
    parser.add_argument("--output_dir", type=str, default="../../data/pca/graphene_AA/stage2_noisy_centering_40",
                        help="输出文件夹路径")
    parser.add_argument("--n_components", type=int, default=1, choices=[1, 3],
                        help="PCA组件数 (1=灰度热力图, 3=RGB)")

    args = parser.parse_args()

    print("=" * 60)
    print("Stage 1 vs Stage 2 Denoising Visualization")
    print("=" * 60)

    # === 加载两个阶段的模型 ===
    stage1_model = load_stage1_model(args.stage1_path)
    stage2_model = load_stage2_model(args.stage2_path)

    print("\n" + "=" * 60)
    print("Starting visualization...")
    print("=" * 60)

    # === 处理图像 ===
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")

    images = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg'))

    if len(images) == 0:
        print(f"警告: 在 {input_dir} 中没有找到图像文件")
        return

    print(f"\n找到 {len(images)} 张图像")

    for img_path in sorted(images):
        visualize_comparison(
            img_path=img_path,
            out_folder=args.output_dir,
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            stage1_name="Stage 1 (Clean)",
            stage2_name="Stage 2 (Denoised)",
            n_components=args.n_components
        )

    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
