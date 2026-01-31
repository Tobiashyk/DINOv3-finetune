import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
from pathlib import Path
from peft import PeftModel

# Set paths
REPO_DIR = '/home/abc/projects/DINOv3/dinov3'
base_weights_path = '/home/abc/projects/DINOv3/dinov3/weight/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
lora_weights_path = '../../dinov3_finetune/student_weights/teacher_student_huge_pos/student_encoder_epoch_40'

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load model ===
print("Loading base DINOv3 model...")
base_model = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=base_weights_path)

print(f"Loading LoRA adapters from {lora_weights_path}...")
try:
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    # model = model.merge_and_unload()  # optional
    print("Successfully loaded Student model with LoRA!")
except Exception as e:
    print(f"Error loading LoRA: {e}")
    print("Falling back to base model (Verify your lora_path!)")
    model = base_model

model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

PATCH_SIZE = 16

def process_image(img_path, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    img = Image.open(img_path).convert('RGB')
    W_orig, H_orig = img.size  # PIL: (W, H)

    # Make size divisible by patch size
    new_w = (W_orig // PATCH_SIZE) * PATCH_SIZE
    new_h = (H_orig // PATCH_SIZE) * PATCH_SIZE
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

    # Tensor (1, 3, H, W)
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        features = model(img_tensor, is_training=True)

    # Patch tokens: (1, num_patches, D) -> (num_patches, D)
    patch_features = features["x_norm_patchtokens"].squeeze(0)
    num_patches, D = patch_features.shape

    # Grid sizes
    h_grid = new_h // PATCH_SIZE
    w_grid = new_w // PATCH_SIZE

    # Sanity check
    if num_patches != h_grid * w_grid:
        raise ValueError(
            f"num_patches mismatch: got {num_patches}, expected {h_grid*w_grid} "
            f"(h_grid={h_grid}, w_grid={w_grid})"
        )

    # ============================================================
    # NEW: Pixel-level PCA (upsample features first, then PCA)
    # ============================================================
    # 1) (num_patches, D) -> (D, h_grid, w_grid)
    feat_map = patch_features.transpose(0, 1).contiguous().view(D, h_grid, w_grid)  # (D, hg, wg)

    # 2) Upsample feature map to pixel resolution BEFORE PCA:
    #    (D, hg, wg) -> (1, D, hg, wg) -> (1, D, new_h, new_w)
    feat_map_4d = feat_map.unsqueeze(0)  # (1, D, hg, wg)
    feat_pix = torch.nn.functional.interpolate(
        feat_map_4d,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=False
    )  # (1, D, new_h, new_w)

    # 3) Prepare PCA input where each pixel is a sample:
    #    (1, D, H, W) -> (H*W, D)
    feat_pix = feat_pix.squeeze(0)  # (D, new_h, new_w)
    X = feat_pix.permute(1, 2, 0).contiguous().view(-1, D)  # (new_h*new_w, D)

    # Move to CPU for sklearn PCA
    X_cpu = X.detach().cpu().numpy()

    # 4) PCA over pixels (n_components=1)
    pca = PCA(n_components=1)
    pca_scores = pca.fit_transform(X_cpu)  # (new_h*new_w, 1)

    # 5) Reshape directly to pixel heatmap (no post-PCA upsampling)
    heatmap = pca_scores.reshape(new_h, new_w)

    # Normalize to [0, 1] for display stability
    hmin, hmax = heatmap.min(), heatmap.max()
    if (hmax - hmin) > 1e-12:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)

    # If resized image differs from original, optionally map back to original size for overlay
    # (This is NOT "PCA after upsampling"; PCA already happened at pixel level of new_h/new_w.
    #  This step only restores to original H_orig/W_orig for consistent visualization.)
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()  # (1,1,new_h,new_w)
    heatmap_full = torch.nn.functional.interpolate(
        heatmap_tensor,
        size=(H_orig, W_orig),
        mode='bicubic',
        align_corners=False
    ).squeeze().numpy()

    # Visualize
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(heatmap_full, cmap='inferno')
    axs[1].set_title("Pixel-level PCA (features upsampled first)")
    axs[1].axis('off')

    # Save
    out_path = os.path.join(out_folder, os.path.basename(img_path))
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# Process Sim_1T
input_dir_1 = Path('../train_pic/test/Sim_1T_1024_test')
if input_dir_1.exists():
    print(f"\n--- Processing {input_dir_1} ---")
    images = list(input_dir_1.glob('*.png')) + list(input_dir_1.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/1T')

# Process Sim_2H
input_dir_2 = Path('../train_pic/test/Sim_2H_1024_test')
if input_dir_2.exists():
    print(f"\n--- Processing {input_dir_2} ---")
    images = list(input_dir_2.glob('*.png')) + list(input_dir_2.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/2H')

print("\nAll Processing complete.")
