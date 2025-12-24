import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
from pathlib import Path

# Set paths
REPO_DIR = '/home/abc/projects/DINOv3/dinov3'
weights_path = '/home/abc/projects/DINOv3/dinov3/weight/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'

# Load DINOv3 model
dinov3_vits16plus = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=weights_path)
dinov3_vits16plus.eval()

# Create output directories
# os.makedirs('./PCA/PCA_1T', exist_ok=True)
# os.makedirs('./PCA/PCA_2H', exist_ok=True)

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def process_image(img_path, out_folder):
    img = Image.open(img_path).convert('RGB')
    W_orig, H_orig = img.size  # Note: PIL is (W, H), but for tensor it's (H, W)
    new_w = (W_orig // 16) * 16
    new_h = (H_orig // 16) * 16
    img_resized = img.resize((new_w, new_h))
    # Transform to tensor
    img_tensor = transform(img_resized).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        features = dinov3_vits16plus(img_tensor, is_training=True)  # dict

    # Get patch tokens
    patch_features = features["x_norm_patchtokens"]  # (1, num_patches, dim)
    patch_features = patch_features.squeeze(0)  # (num_patches, dim)

    # Get grid dimensions
    h_grid = new_h // 16
    w_grid = new_w // 16

    # Reshape to (h_grid * w_grid, dim)
    patch_features_flat = patch_features.view(-1, patch_features.shape[-1])

    # PCA with n_components=1
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(patch_features_flat.cpu().numpy())  # (num_patches, 1)

    # Reshape to (h_grid, w_grid)
    heatmap = pca_result.reshape(h_grid, w_grid)

    # Upsample to original size using bicubic interpolation
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)  # (1, 1, h_grid, w_grid)
    upsampled = torch.nn.functional.interpolate(heatmap_tensor, size=(H_orig, W_orig), mode='bicubic', align_corners=False)
    heatmap_full = upsampled.squeeze().numpy()

    # Visualize: original and heatmap side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[1].imshow(heatmap_full, cmap='inferno')
    axs[1].axis('off')

    # Save
    out_path = os.path.join(out_folder, os.path.basename(img_path))
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

# Process Sim_1T
for img_path in sorted(Path('./Sim_1T_test').glob('*.png')):
    process_image(img_path, './PCA_1T_ori')

# Process Sim_2H
for img_path in sorted(Path('./Sim_2H_test').glob('*.png')):
    process_image(img_path, './PCA_2H_ori')

# for img_path in sorted(Path('./cat').glob('*.png')):
#     process_image(img_path, './PCA_cat')

print("Processing complete.")
