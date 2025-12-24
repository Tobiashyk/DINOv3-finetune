import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
from pathlib import Path

# === [新增] 导入 PEFT 库用于加载 LoRA ===
from peft import PeftModel

SCALE_FACTOR = 4

# === 1. 设置路径 ===
REPO_DIR = '/home/abc/projects/DINOv3/dinov3'
# 原始预训练权重路径 (底座)
base_weights_path = '/home/abc/projects/DINOv3/dinov3/weight/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'

# [新增] 训练好的 Student LoRA 权重文件夹路径
# 这里指向你训练脚本保存的 output 目录，例如 'output/student_final'
lora_weights_path = '../../dinov3_finetune/student_weights/ori_train/student_encoder_epoch_20' 

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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def process_image(img_path, out_folder):
    # 确保输出文件夹存在
    os.makedirs(out_folder, exist_ok=True)
    
    img = Image.open(img_path).convert('RGB')
    W_orig, H_orig = img.size 

    # === [修改点 1] 强制放大图像 (SCALE_FACTOR Upscale) ===
    new_w = (W_orig * SCALE_FACTOR // 16) * 16
    new_h = (H_orig * SCALE_FACTOR // 16) * 16
    
    # 使用双三次插值 (BICUBIC) 进行放大
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # === [修改点 2] 保存放大的图像 ===
    upscaled_filename = f"upscaled_{img_path.name}"
    upscaled_save_path = os.path.join(out_folder, upscaled_filename)
    img_resized.save(upscaled_save_path)

    # 转为 Tensor 并移至 GPU
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # 提取特征
    with torch.no_grad():
        # 注意：这里调用的是 model 而不是 dinov3_vits16plus
        features = model(img_tensor, is_training=True)

    # 获取 Patch Tokens
    patch_features = features["x_norm_patchtokens"]
    patch_features = patch_features.squeeze(0)

    # === [修改点 3] Grid 尺寸计算 ===
    h_grid = new_h // 16
    w_grid = new_w // 16

    # Reshape
    patch_features_flat = patch_features.view(-1, patch_features.shape[-1])

    # 转回 CPU 进行 PCA 计算
    patch_features_cpu = patch_features_flat.cpu().numpy()

    # === PCA 分析 ===
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(patch_features_cpu)

    # Reshape 回热力图网格
    heatmap = pca_result.reshape(h_grid, w_grid)
    
    # [优化] 归一化热力图
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        # === 还原尺寸 ===
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

    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(heatmap_full, cmap='inferno') 
    axs[1].set_title("PCA Heatmap")
    axs[1].axis('off')

    # 保存对比图
    out_path = os.path.join(out_folder, os.path.basename(img_path))
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    print(f"Processed: {img_path.name}")

# === 4. 执行循环 ===

# 处理 Sim_1T
input_dir_1 = Path('../train_pic/Sim_1T_test')
if input_dir_1.exists():
    print(f"\n--- Processing {input_dir_1} ---")
    images = list(input_dir_1.glob('*.png')) + list(input_dir_1.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/1T_huge_finetune') # 修改输出文件夹名称以区分

# 处理 Sim_2H
input_dir_2 = Path('../train_pic/Sim_2H_test')
if input_dir_2.exists():
    print(f"\n--- Processing {input_dir_2} ---")
    images = list(input_dir_2.glob('*.png')) + list(input_dir_2.glob('*.jpg'))
    for img_path in sorted(images):
        process_image(img_path, '../PCA_pic/2H_huge_finetune')

print("\nAll Processing complete.")