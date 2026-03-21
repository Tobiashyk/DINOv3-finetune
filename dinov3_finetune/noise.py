import numpy as np
from scipy.ndimage import zoom, sobel
from scipy.stats import tukeylambda
import matplotlib.pyplot as plt
from PIL import Image

def apply_shot_noise(image, dose=1e3):
    """
    A. Shot Noise (泊松噪声)
    模拟电子计数统计涨落
    
    参数:
        image: 输入图像
        dose: 电子剂量 (electrons/Å²)
    """
    # 归一化图像到 [0, 1]
    image_norm = (image - image.min()) / (image.max() - image.min())
    
    # 计算每个像素的平均电子数
    pixel_area = (0.1) ** 2  # Å²
    mean_electrons = image_norm * dose * pixel_area
    
    # 泊松采样
    noisy_electrons = np.random.poisson(mean_electrons)
    
    # 归一化回原始范围
    noisy_image = noisy_electrons.astype(float)
    noisy_image = (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min())
    noisy_image = noisy_image * (image.max() - image.min()) + image.min()
    
    return noisy_image

def apply_background_noise(image, low_res_size=8, strength=0.2):
    """
    C. Background Noise (背景噪声) - [已修复]
    
    修正点：
    1. 从“加性噪声”改为“乘性噪声” (image * background)。
       物理含义：模拟样品厚度不均导致的透射率变化，或者照明强度的低频波动。
    2. 使用 strength 参数直接控制波动的幅度 (例如 0.2 表示在 0.8~1.2 倍之间波动)。
    """
    h, w = image.shape
    
    # 1. 生成低分辨率的随机起伏场
    # 均值为 1.0，标准差为 strength 的正态分布
    # 这样生成的背景会在 1.0 上下浮动
    low_res_bg = np.random.normal(loc=1.0, scale=strength, size=low_res_size)
    
    # 2. 双三次插值上采样到原图尺寸
    zoom_factor = (h / low_res_size[0], w / low_res_size[1])
    background_field = zoom(low_res_bg, zoom_factor, order=3)
    
    # 裁剪可能多出的边缘（防止舍入误差导致的尺寸不匹配）
    background_field = background_field[:h, :w]
    
    # 3. 乘法叠加
    noisy_image = image * background_field
    
    # 4. 保持数值范围合理 (不小于0)
    return np.clip(noisy_image, 0, None)


def apply_scan_noise(image, k=0.004, b=0.03):
    """
    D. Scan Noise (扫描抖动噪声) - [已修复]
    
    修正点：
    1. 大幅减小了 k 和 b 的值，适配 0-1 的浮点图像。
    2. 现在的 b=0.03 代表约 3% 的基础行间抖动，k=0.004 代表边缘处的额外抖动。
    """
    # 1. 计算垂直梯度
    gradient_y = sobel(image, axis=0)
    gradient_magnitude = np.abs(gradient_y)
    
    # 2. 定义噪声 sigma 映射
    # 这里不需要再乘以 (image.max() - image.min())，直接使用绝对数值控制更精准
    sigma_map = k * gradient_magnitude + b
    
    # 3. 生成行相关噪声 (每行共享相同的随机值)
    # 形状为 (Height,)
    row_noise = np.random.normal(0, 1, image.shape[0])
    
    # 4. 扩展到整个图像 (Height, 1) * (Height, Width)
    # 注意：sigma_map 也是 (Height, Width)，乘法是逐元素的
    scan_noise = row_noise[:, np.newaxis] * sigma_map
    
    return image + scan_noise


def apply_pointwise_noise(image, k=0.08, b=11.01, lam=0.21):
    """
    E. Pointwise Noise (逐点探测器噪声)
    方差与像素强度成正比 (模拟探测器响应非线性)
    
    方法: sigma(x,y) = alpha * sqrt(I(x,y))
    
    参数:
        image: 输入图像
        alpha: 噪声强度系数
    """
    current_max = image.max()
    
    # 2. 计算缩放倍数 (Scale Factor)
    # 逻辑: 论文是基于 255 的，你的图只有 current_max，所以倍数是 255 / current_max
    if current_max == 0:
        factor = 1.0
    else:
        factor = 255.0 / current_max
    
    # 3. 【核心修改】调整参数
    # k 保持不变 (0.08)
    k_scaled = k 
    # b 需要缩小 (例如 11.01 -> 0.0004)
    b_scaled = b / factor
    # 归一化到正值
    sigma_map = k_scaled * image + b_scaled
    
    # 2. 生成 Tukey's Lambda 分布噪声
    # 注意：生成与图像同尺寸的随机数比较慢，scipy的rvs在大数组上可能较慢
    # 也可以预生成一个大的池子或者用近似方法
    noise = tukeylambda.rvs(lam, loc=0, scale=1, size=image.shape)
    
    # 3.不仅要乘以 sigma_map 进行缩放
    noise = noise * sigma_map
    
    return image + noise


def apply_all_noise(image, dose=1e3, bg_size=8, 
                    scan_k=0.02, scan_b=0.01,
                    point_k=0.08, point_b=11.01, point_lam=0.21):
                    
    """
    组合所有噪声模型 (现实模拟)
    """
    noisy = apply_shot_noise(image, dose=dose)
    noisy = apply_background_noise(noisy, low_res_size=bg_size)
    noisy = apply_scan_noise(noisy, k=scan_k, b=scan_b)
    noisy = apply_pointwise_noise(noisy, k=point_k, b=point_b, lam=point_lam)
    
    return noisy


if __name__ == "__main__":
    import os
    
    # 加载测试图像 - 使用项目中的STEM图像
    image_paths = [
        './noise_test/stem_graphene_AA498.png',
        './noise_test/stem_graphene_AA499.png',
        './noise_test/stem_graphene_AA500.png'
    ]
    
    for img_idx, img_path in enumerate(image_paths):
        try:
            # 加载图像并转换为灰度
            img = Image.open(img_path).convert('L')
            image = np.array(img, dtype=float)
            
            # 获取图像所在文件夹
            img_dir = os.path.dirname(img_path)
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            
            # 创建图形显示所有噪声效果
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Noise Comparison - Image {img_idx + 1}', fontsize=16)
            
            # 原图
            axes[0, 0].imshow(image, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Shot Noise
            noisy_shot = apply_shot_noise(image.copy(), dose=2e3)
            axes[0, 1].imshow(noisy_shot, cmap='gray')
            axes[0, 1].set_title('A. Shot Noise (Poisson)')
            axes[0, 1].axis('off')
            
            # Background Noise
            noisy_bg = apply_background_noise(image.copy(), low_res_size=(8,8))
            axes[0, 2].imshow(noisy_bg, cmap='gray')
            axes[0, 2].set_title('B. Background Noise')
            axes[0, 2].axis('off')
            
            # Scan Noise
            noisy_scan = apply_scan_noise(image.copy(), k=0.11, b=0.005)
            axes[1, 0].imshow(noisy_scan, cmap='gray')
            axes[1, 0].set_title('C. Scan Noise (Row-correlated)')
            axes[1, 0].axis('off')
            
            # Pointwise Noise
            noisy_point = apply_pointwise_noise(image.copy(), k=0.08, b=11.01, lam=0.21)
            axes[1, 1].imshow(noisy_point, cmap='gray')
            axes[1, 1].set_title('D. Pointwise Detector Noise')
            axes[1, 1].axis('off')
            
            # All Noise Combined
            noisy_all = apply_all_noise(image.copy(), dose=2e3, bg_size=8,
                                        scan_k=0.11, scan_b=0.005,
                                        point_k=0.08, point_b=11.01, point_lam=0.21)
            axes[1, 2].imshow(noisy_all, cmap='gray')
            axes[1, 2].set_title('E. All Noise Combined')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # 保存到与源图片相同的文件夹
            output_path = os.path.join(img_dir, f'{img_basename}_noise_comparison.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close()
            
        except FileNotFoundError:
            print(f"Error: Cannot find image at {img_path}")
            print("Please save the attached images to the same directory as noise.py")
            print("with filenames: test_image_1.png and test_image_2.png")
    
    print("\nNoise generation complete!")