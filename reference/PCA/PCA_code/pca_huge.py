import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
from pathlib import Path

# === 1. 设置路径 ===
REPO_DIR = '/home/abc/projects/DINOv3/dinov3'
weights_path = '/home/abc/projects/DINOv3/dinov3/weight/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 2. 加载模型 ===
dinov3_vits16plus = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=weights_path)
dinov3_vits16plus.to(device)
dinov3_vits16plus.eval()

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

def process_image(img_path, out_folder):
    # 确保输出文件夹存在
    os.makedirs(out_folder, exist_ok=True)
    
    img = Image.open(img_path).convert('RGB')

    # 使用双三次插值 (BICUBIC) 进行放大，保证边缘平滑
    img_resized = transform_resize(img)
    W_orig, H_orig = img.size
    new_w, new_h = img_resized.size
    # 转为 Tensor 并移至 GPU
    img_tensor = transform_tensor(img_resized).unsqueeze(0).to(device)

    # 提取特征
    with torch.no_grad():
        features = dinov3_vits16plus(img_tensor, is_training=True)

    # 获取 Patch Tokens
    patch_features = features["x_norm_patchtokens"]
    patch_features = patch_features.squeeze(0)

    # Reshape
    patch_features_flat = patch_features.view(-1, patch_features.shape[-1])

    # === PCA 分析 ===
    # 依然建议：如果 n_components=1 效果不好，可以试试改大一点看看 component 2/3
    pca = PCA(n_components=1) 
    
    # 转回 CPU 进行 PCA 计算
    patch_features_cpu = patch_features_flat.cpu().numpy()
    pca_result = pca.fit_transform(patch_features_cpu)

    # Reshape 回热力图网格
    heatmap = pca_result.reshape(new_h // 16, new_w // 16)
    
    # [优化] 归一化热力图 (让对比度拉满)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # === 还原尺寸 ===
    # 注意：这里我们将热力图放大回【原图的原始尺寸】(H_orig, W_orig)
    # 这样方便和原图做对比
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    upsampled = torch.nn.functional.interpolate(
        heatmap_tensor, 
        size=(H_orig, W_orig), 
        mode='bicubic', 
        align_corners=False
    )
    heatmap_full = upsampled.squeeze().numpy()

    # === 可视化 ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # 左图：原始图像 (未放大的)
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    # 右图：基于放大图计算出的 PCA 热力图
    axs[1].imshow(heatmap_full, cmap='inferno')
    axs[1].set_title("PCA Heatmap (from 2x Upscaled Input)")
    axs[1].axis('off')

    # 保存对比图
    out_path = os.path.join(out_folder, os.path.basename(img_path))
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# === 4. 执行循环 ===

# 处理 
# input_dir_1 = Path('../train_pic/test/lpy_1024')
# if input_dir_1.exists():
#     print(f"\n--- Processing {input_dir_1} ---")
#     images = list(input_dir_1.glob('*.png')) + list(input_dir_1.glob('*.jpg'))
#     for img_path in sorted(images):
#         process_image(img_path, '../PCA_pic/lpy_1024') # 修改输出文件夹名称以区分

# 处理 Sim_1T
input_dir_1 = Path('../train_pic/test/Sim_1T_256_test')
if input_dir_1.exists():
    print(f"\n--- Processing {input_dir_1} ---")
    images = list(input_dir_1.glob('*.png')) + list(input_dir_1.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/1T')

# 处理 Sim_2H
input_dir_2 = Path('../train_pic/test/Sim_2H_256_test')
if input_dir_2.exists():
    print(f"\n--- Processing {input_dir_2} ---")
    images = list(input_dir_2.glob('*.png')) + list(input_dir_2.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/2H')

print("\nAll Processing complete.")