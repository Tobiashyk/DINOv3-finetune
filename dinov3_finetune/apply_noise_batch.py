"""
批量对STEM图像添加多种噪声
从 PCA/train_pic 中的 images_noisy 文件夹读取图像，
应用随机强度的四类噪声，保存到 images_noisy_all 文件夹，
并记录噪声参数到 labels_noisy_all.json
"""

import numpy as np
import os
import json
from PIL import Image
from scipy.ndimage import zoom, sobel
from scipy.stats import tukeylambda
from tqdm import tqdm
import random


def apply_shot_noise(image, dose=1e3):
    """Shot Noise (泊松噪声)"""
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    pixel_area = (0.1) ** 2
    mean_electrons = image_norm * dose * pixel_area
    noisy_electrons = np.random.poisson(mean_electrons)
    noisy_image = noisy_electrons.astype(float)
    noisy_image = (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min() + 1e-8)
    noisy_image = noisy_image * (image.max() - image.min()) + image.min()
    return noisy_image


def apply_background_noise(image, low_res_size=(8, 8), strength=0.2):
    """Background Noise (背景噪声) - 乘性噪声"""
    h, w = image.shape
    low_res_bg = np.random.normal(loc=1.0, scale=strength, size=low_res_size)
    zoom_factor = (h / low_res_size[0], w / low_res_size[1])
    background_field = zoom(low_res_bg, zoom_factor, order=3)
    background_field = background_field[:h, :w]
    noisy_image = image * background_field
    return np.clip(noisy_image, 0, None)


def apply_scan_noise(image, k=0.002, b=0.008):
    """Scan Noise (扫描抖动噪声)"""
    gradient_y = sobel(image, axis=0)
    gradient_magnitude = np.abs(gradient_y)
    sigma_map = k * gradient_magnitude + b
    sigma_map = sigma_map * (image.max() - image.min() + 1e-8)
    row_noise = np.random.normal(0, 1, image.shape[0])
    scan_noise = row_noise[:, np.newaxis] * sigma_map
    return image + scan_noise


def apply_pointwise_noise(image, k=0.08, b=11.01, lam=0.21):
    """Pointwise Noise (逐点探测器噪声)"""
    current_max = image.max()
    if current_max == 0:
        factor = 1.0
    else:
        factor = 255.0 / current_max
    k_scaled = k
    b_scaled = b / factor
    sigma_map = k_scaled * image + b_scaled
    noise = tukeylambda.rvs(lam, loc=0, scale=1, size=image.shape)
    noise = noise * sigma_map
    return image + noise


def apply_all_noise_random(image):
    """
    应用所有噪声，参数随机化
    
    返回: (noisy_image, noise_params)
    """
    # 随机噪声参数
    dose = random.uniform(3000, 5000)  # Shot noise
    bg_strength = random.uniform(0.25, 0.45)  # Background noise strength
    bg_grid_size = random.randint(8, 10)  # Background grid size
    scan_b = random.uniform(0.001, 0.0012)  # Scan noise base
    scan_k = random.uniform(0.0005, 0.0006)  # Scan noise gradient
    point_k = random.uniform(0.01, 0.025)  # Pointwise signal
    point_b = random.uniform(0.005, 0.015)  # Pointwise base

    # 记录参数
    noise_params = {
        'shot_noise_dose': dose,
        'background_strength': bg_strength,
        'background_grid_size': bg_grid_size,
        'scan_noise_base': scan_b,
        'scan_noise_gradient': scan_k,
        'pointwise_signal': point_k,
        'pointwise_base': point_b
    }
    
    # 依次应用噪声
    noisy = apply_shot_noise(image.copy(), dose=dose)
    noisy = apply_background_noise(noisy, low_res_size=(bg_grid_size, bg_grid_size), 
                                    strength=bg_strength)
    noisy = apply_scan_noise(noisy, k=scan_k, b=scan_b)
    noisy = apply_pointwise_noise(noisy, k=point_k, b=point_b, lam=0.21)
    
    return noisy, noise_params


def process_dataset(input_dir, output_dir, labels_file, num_images=500):
    """
    处理一个数据集文件夹
    
    参数:
        input_dir: 输入文件夹路径 (images_noisy)
        output_dir: 输出文件夹路径 (images_noisy_all)
        labels_file: 标签文件路径 (labels_noisy_all.json)
        num_images: 处理图像数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    # 限制数量
    if len(image_files) > num_images:
        image_files = random.sample(image_files, num_images)
    
    print(f"Processing {len(image_files)} images from {input_dir}")
    
    # 记录所有图像的噪声参数
    labels_data = {}
    
    # 批量处理
    for img_file in tqdm(image_files, desc="Adding noise"):
        try:
            # 加载图像
            img_path = os.path.join(input_dir, img_file)
            img = Image.open(img_path).convert('L')
            image = np.array(img, dtype=float)
            
            # 应用随机噪声
            noisy_image, noise_params = apply_all_noise_random(image)
            
            # 转换回uint8并保存
            noisy_image_uint8 = np.clip(noisy_image, 0, 255).astype(np.uint8)
            output_path = os.path.join(output_dir, img_file)
            Image.fromarray(noisy_image_uint8).save(output_path)
            
            # 记录噪声参数
            labels_data[img_file] = noise_params
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    # 保存标签文件
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(labels_data)} images to {output_dir}")
    print(f"Labels saved to {labels_file}")


def main():
    # 基础路径
    base_dir = r"d:\Projects\DINOv3-finetune\PCA\train_pic"
    
    # 数据集配置
    datasets = [
        # {
        #     'name': 'graphene_stem_dataset_AA',
        #     'input': os.path.join(base_dir, 'graphene_stem_dataset_AA', 'images_noisy'),
        #     'output': os.path.join(base_dir, 'graphene_stem_dataset_AA', 'images_noisy_all'),
        #     'labels': os.path.join(base_dir, 'graphene_stem_dataset_AA', 'labels_noisy_all.json')
        # },
        # {
        #     'name': 'graphene_stem_dataset_AB',
        #     'input': os.path.join(base_dir, 'graphene_stem_dataset_AB', 'images_noisy'),
        #     'output': os.path.join(base_dir, 'graphene_stem_dataset_AB', 'images_noisy_all'),
        #     'labels': os.path.join(base_dir, 'graphene_stem_dataset_AB', 'labels_noisy_all.json')
        # }

        # {
        #     'name': 'MoS2_stem_dataset_2H',
        #     'input': os.path.join(base_dir, 'MoS2_stem_dataset_2H', 'images_noisy'),
        #     'output': os.path.join(base_dir, 'MoS2_stem_dataset_2H', 'images_noisy_all'),
        #     'labels': os.path.join(base_dir, 'MoS2_stem_dataset_2H', 'labels_noisy_all.json')
        # },
        # {
        #     'name': 'MoS2_stem_dataset_1T',
        #     'input': os.path.join(base_dir, 'MoS2_stem_dataset_1T', 'images_noisy'),
        #     'output': os.path.join(base_dir, 'MoS2_stem_dataset_1T', 'images_noisy_all'),
        #     'labels': os.path.join(base_dir, 'MoS2_stem_dataset_1T', 'labels_noisy_all.json')
        # }
        {
            'name': 'paper_stem',
            'input': os.path.join(base_dir, 'paper_stem', 'images_noisy'),
            'output': os.path.join(base_dir, 'paper_stem', 'images_noisy_all'),
            'labels': os.path.join(base_dir, 'paper_stem', 'labels_noisy_all.json')
        }
    ]
    
    # 处理每个数据集
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset['name']}")
        print(f"{'='*60}")
        
        if not os.path.exists(dataset['input']):
            print(f"Warning: Input directory not found: {dataset['input']}")
            continue
        
        process_dataset(
            input_dir=dataset['input'],
            output_dir=dataset['output'],
            labels_file=dataset['labels'],
            num_images=500
        )
    
    print(f"\n{'='*60}")
    print("All datasets processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
