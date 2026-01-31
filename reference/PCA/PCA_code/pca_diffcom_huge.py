import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
from pathlib import Path

SCALE_FACTOR = 4
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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def process_image(img_path, out_folder):
    # 确保输出文件夹存在
    os.makedirs(out_folder, exist_ok=True)
    
    img = Image.open(img_path).convert('RGB')
    W_orig, H_orig = img.size 

    # === [修改点 1] 强制放大图像 (2x Upscale) ===
    # 乘以 2，然后确保是 16 的倍数
    # 这样原本 10px 的原子会变成 20px，占据 1-2 个 Patch
    new_w = (W_orig * SCALE_FACTOR // 16) * 16
    new_h = (H_orig * SCALE_FACTOR // 16) * 16
    
    # 使用双三次插值 (BICUBIC) 进行放大，保证边缘平滑
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # === [修改点 2] 保存放大的图像 ===
    # 这样你可以直观地看到喂给模型的图长什么样
    upscaled_filename = f"upscaled_{img_path.name}"
    upscaled_save_path = os.path.join(out_folder, upscaled_filename)
    img_resized.save(upscaled_save_path)
    print(f"Saved upscaled image: {upscaled_save_path}")

    # 转为 Tensor 并移至 GPU
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # 提取特征
    with torch.no_grad():
        features = dinov3_vits16plus(img_tensor, is_training=True)

    # 获取 Patch Tokens
    patch_features = features["x_norm_patchtokens"]
    patch_features = patch_features.squeeze(0)

    # === [修改点 3] Grid 尺寸计算 ===
    # 注意：这里必须用放大后的 new_h / new_w 来计算
    h_grid = new_h // 16
    w_grid = new_w // 16

    # Reshape
    patch_features_flat = patch_features.view(-1, patch_features.shape[-1])

    # 转回 CPU 进行 PCA 计算
    patch_features_cpu = patch_features_flat.cpu().numpy()

    # === PCA 分析 ===
    # 依然建议：如果 n_components=1 效果不好，可以试试改大一点看看 component 2/3
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(patch_features_cpu)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(4):
    # Reshape 回热力图网格
        component = pca_result[:, i]
        heatmap = component.reshape(h_grid, w_grid)
    
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
        axs[i].imshow(heatmap_full, cmap='inferno') # 或者试用 'gray'
        axs[i].set_title(f"PCA Component {i+1}")
        axs[i].axis('off')

    # 保存对比图
    out_path = os.path.join(out_folder, os.path.basename(img_path))
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    print(f"Processed: {img_path.name} (Upscaled saved as {upscaled_filename})")

# === 4. 执行循环 ===

# 处理 Sim_1T
input_dir_1 = Path('../train_pic/Sim_1T_test')
if input_dir_1.exists():
    print(f"\n--- Processing {input_dir_1} ---")
    # 支持多种格式
    images = list(input_dir_1.glob('*.png')) + list(input_dir_1.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/1T_diffcom_huge')

# 处理 Sim_2H
input_dir_2 = Path('../train_pic/Sim_2H_test')
if input_dir_2.exists():
    print(f"\n--- Processing {input_dir_2} ---")
    images = list(input_dir_2.glob('*.png')) + list(input_dir_2.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/2H_diffcom_huge')

print("\nAll Processing complete.")