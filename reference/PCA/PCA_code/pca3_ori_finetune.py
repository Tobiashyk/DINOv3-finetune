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
lora_weights_path = '../../dinov3_finetune/student_weights/teacher_student_huge/student_encoder_epoch_40'

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

PATCH_SIZE = 16

def minmax_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize array to [0,1]."""
    x_min = x.min()
    x_max = x.max()
    if (x_max - x_min) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)

def process_image(img_path, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    img = Image.open(img_path).convert('RGB')
    W_orig, H_orig = img.size

    # Ensure divisible by patch size
    new_w = (W_orig // PATCH_SIZE) * PATCH_SIZE
    new_h = (H_orig // PATCH_SIZE) * PATCH_SIZE
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

    img_tensor = transform(img_resized).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        features = model(img_tensor, is_training=True)

    patch_features = features["x_norm_patchtokens"].squeeze(0)  # (num_patches, D)

    h_grid = new_h // PATCH_SIZE
    w_grid = new_w // PATCH_SIZE

    patch_features_flat = patch_features.view(-1, patch_features.shape[-1])  # (num_patches, D)
    patch_features_cpu = patch_features_flat.detach().cpu().numpy()

    # =========================
    # PCA 3-channel (RGB) HERE
    # =========================
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(patch_features_cpu)  # (num_patches, 3)

    # Reshape to patch grid with 3 channels: (h_grid, w_grid, 3)
    heatmap3 = pca_result.reshape(h_grid, w_grid, 3)

    # Normalize each channel to [0,1] for visualization
    heatmap3_norm = np.zeros_like(heatmap3, dtype=np.float32)
    for c in range(3):
        heatmap3_norm[..., c] = minmax_norm(heatmap3[..., c])

    # Upsample each channel to original size
    # Convert to tensor shape (1, 3, h_grid, w_grid) for interpolate
    heatmap3_tensor = torch.from_numpy(heatmap3_norm).permute(2, 0, 1).unsqueeze(0)  # (1, 3, h_grid, w_grid)
    upsampled = torch.nn.functional.interpolate(
        heatmap3_tensor,
        size=(H_orig, W_orig),
        mode='bicubic',
        align_corners=False
    )  # (1, 3, H_orig, W_orig)

    # Back to numpy image (H, W, 3)
    heatmap3_full = upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy()
    heatmap3_full = np.clip(heatmap3_full, 0.0, 1.0)

    # Visualize: original and PCA3 RGB side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(heatmap3_full)  # already RGB in [0,1]
    axs[1].set_title("PCA RGB (3 components on patch tokens)")
    axs[1].axis('off')

    # Save
    out_path = os.path.join(out_folder, os.path.basename(img_path))
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved: {out_path}")

# Process Sim_1T
input_dir_1 = Path('../train_pic/test/Sim_1T_2048_test')
if input_dir_1.exists():
    print(f"\n--- Processing {input_dir_1} ---")
    images = list(input_dir_1.glob('*.png')) + list(input_dir_1.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/1T')

# Process Sim_2H
input_dir_2 = Path('../train_pic/test/Sim_2H_2048_test')
if input_dir_2.exists():
    print(f"\n--- Processing {input_dir_2} ---")
    images = list(input_dir_2.glob('*.png')) + list(input_dir_2.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/2H')

print("\nAll Processing complete.")
