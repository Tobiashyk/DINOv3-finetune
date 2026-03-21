# import torch
# import numpy as np
# from sklearn.decomposition import PCA


# def pca_transform_features(features: torch.Tensor, patch_h: int, patch_w: int, n_components: int = 1) -> np.ndarray:
#     """
#     Apply PCA transformation to features and normalize the result.

#     Args:
#         features: Tensor of shape (B, N, F) where B is batch size, N is number of patches, F is feature dimension
#         patch_h: Height of the patch grid
#         patch_w: Width of the patch grid
#         n_components: Number of PCA components (default: 1)

#     Returns:
#         Normalized PCA features of shape (B, patch_h, patch_w, n_components)
#     """
#     B, N, F = features.shape

#     # Apply PCA transformation
#     pca = PCA(n_components=n_components)
#     pca_features = pca.fit_transform(features.reshape(B * N, F).cpu().numpy())
#     pca_rgb = pca_features.reshape(B, patch_h, patch_w, n_components)

#     # Normalize to [0, 1]
#     pca_rgb_norm = (pca_rgb - pca_rgb.min()) / (pca_rgb.max() - pca_rgb.min())

#     return pca_rgb_norm
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from pathlib import Path
import argparse
import numpy as np

# === [新增] 导入 PEFT 库用于加载 LoRA ===
from peft import PeftModel



# === 1. 设置路径 ===
REPO_DIR = '../../dinov3'
# 原始预训练权重路径 (底座)
parser = argparse.ArgumentParser(description="PCA Heatmap with LoRA Finetuned DINOv3")
parser.add_argument("--weight_path", type=str, default="improved_teacher_student_seed")
parser.add_argument("--num_epochs", type=str, default="40")
parser.add_argument("--factor", type=int, default=1)

args = parser.parse_args()
base_weights_path = '../../weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
SCALE_FACTOR = args.factor

# [新增] 训练好的 Student LoRA 权重文件夹路径
# 这里指向你训练脚本保存的 output 目录，例如 'output/student_final'
lora_weights_path = '../../student_weights/' + args.weight_path + '/student_encoder_epoch_' + args.num_epochs

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 2. 加载模型 (修改部分) ===
print("Loading base DINOv3 model...")
# 1. 先加载原始 DINOv3 底座
base_model = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=base_weights_path)

# 2. 加载 LoRA 权重并合并
print(f"Loading LoRA adapters from {lora_weights_path}...")
try:
    # 将 LoRA 挂载到底座上
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    
    # [可选] 如果你想把 LoRA 权重彻底合并进底座以加快推理速度（不做也可以）
    # model = model.merge_and_unload()
    
    print("Successfully loaded Student model with LoRA!")
except Exception as e:
    print(f"Error loading LoRA: {e}")
    print("Falling back to base model (Verify your lora_path!)")
    model = base_model

model.to(device)
model.eval()

# === 3. 定义预处理 ===
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_resize = transforms.RandomResizedCrop(
            size=(1024, 1024),  # Original size,
            scale=(1.0, 1.0),
            ratio=(1.0, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR
        )

def process_image_comparison_component3(img_path, out_folder, base_model, finetuned_model):
    os.makedirs(out_folder, exist_ok=True)
    
    img = Image.open(img_path).convert('RGB')
    img = transform_resize(img)
    W_orig, H_orig = img.size 
    
    new_w = (W_orig * SCALE_FACTOR // 16) * 16
    new_h = (H_orig * SCALE_FACTOR // 16) * 16
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    img_tensor = transform_tensor(img_resized).unsqueeze(0).to(device)
    
    h_grid = new_h // 16
    w_grid = new_w // 16
    
    # === 提取两个模型的特征 ===
    with torch.no_grad():
        # 原始模型
        base_features = base_model.forward_features(img_tensor)
        base_patches = base_features["x_norm_patchtokens"].squeeze(0).cpu().numpy()
        
        # 微调模型
        finetuned_features = finetuned_model.forward_features(img_tensor)
        finetuned_patches = finetuned_features["x_norm_patchtokens"].squeeze(0).cpu().numpy()
    
    # === PCA 分析 (原始模型) ===
    pca_base = PCA(n_components=3)
    pca_result_base = pca_base.fit_transform(base_patches)
    heatmap_base = pca_result_base.reshape(h_grid, w_grid, 3)
    p5, p95 = np.percentile(heatmap_base, [5, 95])
    heatmap_base = np.clip(heatmap_base, p5, p95)
    heatmap_base = (heatmap_base - heatmap_base.min()) / (heatmap_base.max() - heatmap_base.min())
    
    # === PCA 分析 (微调模型) ===
    pca_finetuned = PCA(n_components=3)
    pca_result_finetuned = pca_finetuned.fit_transform(finetuned_patches)
    heatmap_finetuned = pca_result_finetuned.reshape(h_grid, w_grid, 3)
    p5, p95 = np.percentile(heatmap_finetuned, [5, 95])
    heatmap_finetuned = np.clip(heatmap_finetuned, p5, p95)
    heatmap_finetuned = (heatmap_finetuned - heatmap_finetuned.min()) / (heatmap_finetuned.max() - heatmap_finetuned.min())
    
    # === 上采样到原图尺寸 ===
    def upsample_heatmap(heatmap_rgb):
        heatmap_tensor = torch.from_numpy(heatmap_rgb).permute(2, 0, 1).unsqueeze(0).float()
        upsampled = torch.nn.functional.interpolate(
            heatmap_tensor, 
            size=(H_orig, W_orig), 
            mode='bicubic', 
            align_corners=False
        )
        result = upsampled.squeeze().permute(1, 2, 0).numpy()
        return np.clip(result, 0, 1)  # (H, W, 3)
    
    heatmap_base_full = upsample_heatmap(heatmap_base)
    heatmap_finetuned_full = upsample_heatmap(heatmap_finetuned)
    
    # === 三图可视化 ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    axs[1].imshow(heatmap_base_full)  # 移除 cmap
    axs[1].set_title("Base Model PCA")
    axs[1].axis('off')
    
    axs[2].imshow(heatmap_finetuned_full)  # 移除 cmap
    axs[2].set_title("Finetuned Model PCA")
    axs[2].axis('off')
    
    out_path = os.path.join(out_folder, os.path.basename(img_path))
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Processed: {img_path.name}")

def process_image_comparison_component1(img_path, out_folder, base_model, finetuned_model):
    os.makedirs(out_folder, exist_ok=True)
    
    img = Image.open(img_path).convert('RGB')
    img = transform_resize(img)
    W_orig, H_orig = img.size 
    
    new_w = (W_orig * SCALE_FACTOR // 16) * 16
    new_h = (H_orig * SCALE_FACTOR // 16) * 16
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    img_tensor = transform_tensor(img_resized).unsqueeze(0).to(device)
    
    h_grid = new_h // 16
    w_grid = new_w // 16
    
    # === 提取两个模型的特征 ===
    with torch.no_grad():
        # 原始模型
        base_features = base_model.forward_features(img_tensor)
        base_patches = base_features["x_norm_patchtokens"].squeeze(0).cpu().numpy()
        
        # 微调模型
        finetuned_features = finetuned_model.forward_features(img_tensor)
        finetuned_patches = finetuned_features["x_norm_patchtokens"].squeeze(0).cpu().numpy()
    
    # === PCA 分析 (原始模型) ===
    pca_base = PCA(n_components=1)
    pca_result_base = pca_base.fit_transform(base_patches)
    heatmap_base = pca_result_base.reshape(h_grid, w_grid)
    p5, p95 = np.percentile(heatmap_base, [5, 95])
    heatmap_base = np.clip(heatmap_base, p5, p95)
    heatmap_base = (heatmap_base - heatmap_base.min()) / (heatmap_base.max() - heatmap_base.min())

# === PCA 分析 (微调模型) ===
    pca_finetuned = PCA(n_components=1)
    pca_result_finetuned = pca_finetuned.fit_transform(finetuned_patches)
    heatmap_finetuned = pca_result_finetuned.reshape(h_grid, w_grid)
    p5, p95 = np.percentile(heatmap_finetuned, [5, 95])
    heatmap_finetuned = np.clip(heatmap_finetuned, p5, p95)
    heatmap_finetuned = (heatmap_finetuned - heatmap_finetuned.min()) / (heatmap_finetuned.max() - heatmap_finetuned.min())
    
    # === 上采样到原图尺寸 ===
    def upsample_heatmap(heatmap_gray):
        heatmap_tensor = torch.from_numpy(heatmap_gray).unsqueeze(0).unsqueeze(0).float()  # ✅ [1,1,H,W]
        upsampled = torch.nn.functional.interpolate(
            heatmap_tensor, 
            size=(H_orig, W_orig), 
            mode='bicubic', 
            align_corners=False
        )
        result = upsampled.squeeze().numpy()
        return np.clip(result, 0, 1)
    
    heatmap_base_full = upsample_heatmap(heatmap_base)
    heatmap_finetuned_full = upsample_heatmap(heatmap_finetuned)
    
    # === 三图可视化 ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(heatmap_base_full, cmap='inferno')
    axs[1].set_title("Base Model PCA")
    axs[1].axis('off')

    axs[2].imshow(heatmap_finetuned_full, cmap='inferno')
    axs[2].set_title("Finetuned Model PCA")
    axs[2].axis('off')
    
    out_path = os.path.join(out_folder, os.path.basename(img_path))
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Processed: {img_path.name}")
# === 修改主循环 ===
# 重新加载独立的原始模型（不带 LoRA）
print("Loading separate base model for comparison...")
base_model_copy = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=base_weights_path)
base_model_copy.to(device)
base_model_copy.eval()

finetuned_model = model  # 微调后的模型（带 LoRA）

input_dir_2 = Path('../../data/test_pic/Sim_2H_1024_test')
if input_dir_2.exists():
    print(f"\n--- Processing {input_dir_2} ---")
    images = list(input_dir_2.glob('*.png')) + list(input_dir_2.glob('*.jpg'))
    for img_path in sorted(images):
        process_image_comparison_component1(img_path, '../../data/pca/Sim_2H_1024', base_model_copy, finetuned_model)

print("\nAll Processing complete.")