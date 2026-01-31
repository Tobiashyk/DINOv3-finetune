import torch
import torch.nn.functional as F
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

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading base DINOv3 model...")
base_model = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=base_weights_path)
model = base_model

model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

PATCH_SIZE = 16
SHIFT = 4
SHIFTS = [(0, 0), (SHIFT, 0), (0, SHIFT), (SHIFT, SHIFT)]  # (dx, dy)

def shift_tensor_keep_size(x: torch.Tensor, dx: int, dy: int, mode: str = "reflect") -> torch.Tensor:
    """
    Shift tensor content by (dx, dy) pixels while keeping same H,W.
    x: (B, C, H, W)
    dx > 0 shifts right, dy > 0 shifts down.
    """
    B, C, H, W = x.shape
    pad_l = max(dx, 0)
    pad_r = max(-dx, 0)
    pad_t = max(dy, 0)
    pad_b = max(-dy, 0)

    x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)

    # Crop back to (H, W)
    x_crop = x_pad[:, :, pad_t:pad_t + H, pad_l:pad_l + W]
    return x_crop

def minmax_norm(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    amin, amax = arr.min(), arr.max()
    if (amax - amin) < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - amin) / (amax - amin)).astype(np.float32)

def process_image(img_path, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    img = Image.open(img_path).convert('RGB')
    W_orig, H_orig = img.size  # PIL (W,H)

    # make divisible by patch size
    new_w = (W_orig // PATCH_SIZE) * PATCH_SIZE
    new_h = (H_orig // PATCH_SIZE) * PATCH_SIZE
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

    # tensor (1,3,new_h,new_w)
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # grid sizes for patch tokens
    h_grid = new_h // PATCH_SIZE
    w_grid = new_w // PATCH_SIZE

    feat_sum = None
    D = None

    with torch.no_grad():
        for (dx, dy) in SHIFTS:
            # 1) shift input image tensor (keep 256x256 etc.)
            x_shift = shift_tensor_keep_size(img_tensor, dx=dx, dy=dy, mode="reflect")

            # 2) forward -> patch tokens
            features = model(x_shift, is_training=True)
            patch_tokens = features["x_norm_patchtokens"].squeeze(0)  # (num_patches, D)
            num_patches, D_local = patch_tokens.shape

            if num_patches != h_grid * w_grid:
                raise ValueError(
                    f"num_patches mismatch: got {num_patches}, expected {h_grid*w_grid} "
                    f"(h_grid={h_grid}, w_grid={w_grid})"
                )

            if D is None:
                D = D_local

            # 3) (num_patches, D) -> (1, D, h_grid, w_grid)
            feat_low = patch_tokens.transpose(0, 1).contiguous().view(1, D, h_grid, w_grid)

            # 4) upsample features to pixel space BEFORE alignment: (1, D, new_h, new_w)
            feat_pix = F.interpolate(feat_low, size=(new_h, new_w), mode="bicubic", align_corners=False)

            # 5) reverse shift to align back to original coordinate frame
            feat_aligned = shift_tensor_keep_size(feat_pix, dx=-dx, dy=-dy, mode="reflect")

            feat_sum = feat_aligned if feat_sum is None else (feat_sum + feat_aligned)

    # 6) average fused feature field: (1, D, new_h, new_w)
    feat_avg = feat_sum / float(len(SHIFTS))

    # 7) PCA on pixels: (new_h*new_w, D) -> PCA3
    X = feat_avg.squeeze(0).permute(1, 2, 0).contiguous().view(-1, D)  # (N, D)
    X_cpu = X.detach().cpu().numpy()

    pca = PCA(n_components=3)
    pca_scores = pca.fit_transform(X_cpu)  # (N, 3)

    # reshape to (H, W, 3)
    pca_img = pca_scores.reshape(new_h, new_w, 3)

    # normalize each PCA channel independently to [0,1]
    pca_img_norm = np.zeros_like(pca_img, dtype=np.float32)
    for c in range(3):
        pca_img_norm[..., c] = minmax_norm(pca_img[..., c])

    # resize back to original size if needed (only for visualization consistency)
    pca_tensor = torch.from_numpy(pca_img_norm).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,new_h,new_w)
    pca_full = F.interpolate(
        pca_tensor,
        size=(H_orig, W_orig),
        mode="bicubic",
        align_corners=False
    ).squeeze(0).permute(1, 2, 0).numpy()

    pca_full = np.clip(pca_full, 0.0, 1.0)

    # Visualize
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title("Original")

    axs[1].imshow(pca_full)  # RGB PCA
    axs[1].axis('off')
    axs[1].set_title(f"Shift-fused Pixel PCA3 (shift={SHIFT}, {len(SHIFTS)} views)")

    out_path = os.path.join(out_folder, os.path.basename(img_path))
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# 处理 Sim_1T
input_dir_1 = Path('../train_pic/cat')
if input_dir_1.exists():
    print(f"\n--- Processing {input_dir_1} ---")
    images = list(input_dir_1.glob('*.png')) + list(input_dir_1.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/cat')

# 处理 Sim_2H
# input_dir_2 = Path('../train_pic/test/Sim_2H_512_test')
# if input_dir_2.exists():
#     print(f"\n--- Processing {input_dir_2} ---")
#     images = list(input_dir_2.glob('*.png')) + list(input_dir_2.glob('*.jpg'))
#     for img_path in sorted(images):
#         process_image(img_path, '../PCA_pic/2H')

print("\nAll Processing complete.")
