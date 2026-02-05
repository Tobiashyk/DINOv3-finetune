"""
DINOv3 图像处理完整示例
展示如何使用不同的数据增强策略
"""

import sys
import os

# 添加项目路径到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dinov3'))

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 方式1: 使用完整 DINOv3 的 Multi-Crop 增强
try:
    from dinov3.dinov3.data.augmentations import DataAugmentationDINO
    FULL_DINOV3_AVAILABLE = True
except ImportError:
    print("警告: 无法导入完整 DINOv3 模块，将跳过相关演示")
    FULL_DINOV3_AVAILABLE = False

# 方式2: 使用简化的 Mini-DINO 增强
from src.utils.image_process import base_transform, advance_transform

# 方式3: 使用训练脚本中的增强
from torchvision import transforms


def demo_full_dinov3_augmentation():
    """演示完整 DINOv3 的 Multi-Crop 数据增强"""
    if not FULL_DINOV3_AVAILABLE:
        print("跳过完整 DINOv3 演示（模块未加载）")
        return None

    print("="*60)
    print("方式1: 完整 DINOv3 Multi-Crop 增强")
    print("="*60)

    # 创建增强器
    augmentation = DataAugmentationDINO(
        global_crops_scale=(0.32, 1.0),      # 全局裁剪缩放范围
        local_crops_scale=(0.05, 0.32),      # 局部裁剪缩放范围
        local_crops_number=8,                 # 8个局部裁剪
        global_crops_size=256,                # 全局裁剪尺寸
        local_crops_size=112,                 # 局部裁剪尺寸
    )

    # 加载测试图像
    image = Image.open('data/koala_test/koala_0.png').convert('RGB')
    print(f"原始图像尺寸: {image.size}")

    # 应用增强
    output = augmentation(image)

    print(f"\n增强结果:")
    print(f"  - 全局裁剪数量: {len(output['global_crops'])}")
    print(f"  - 全局裁剪形状: {output['global_crops'][0].shape}")
    print(f"  - 局部裁剪数量: {len(output['local_crops'])}")
    print(f"  - 局部裁剪形状: {output['local_crops'][0].shape}")

    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # 显示2个全局裁剪
    for i in range(2):
        img = output['global_crops'][i]
        # 反归一化以便显示
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Global Crop {i+1}')
        axes[0, i].axis('off')

    # 显示前3个局部裁剪
    for i in range(3):
        img = output['local_crops'][i]
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[0, i+2].imshow(img)
        axes[0, i+2].set_title(f'Local Crop {i+1}')
        axes[0, i+2].axis('off')

    # 显示更多局部裁剪
    for i in range(5):
        if i + 3 < len(output['local_crops']):
            img = output['local_crops'][i+3]
            img = img.permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f'Local Crop {i+4}')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/dinov3_multicrop_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: outputs/dinov3_multicrop_demo.png")
    plt.close()

    return output


def demo_mini_dino_augmentation():
    """演示简化的 Mini-DINO 数据增强"""
    print("\n" + "="*60)
    print("方式2: 简化 Mini-DINO 增强")
    print("="*60)

    # 创建两种变换
    base_tf = base_transform(resize_w=224, resize_h=224)
    advance_tf = advance_transform(resize_w=224, resize_h=224)

    # 加载测试图像
    image = Image.open('data/koala_test/koala_0.png').convert('RGB')

    # 应用变换
    base_img = base_tf(image)
    advance_img = advance_tf(image)

    print(f"\n增强结果:")
    print(f"  - 基础变换输出形状: {base_img.shape}")
    print(f"  - 高级变换输出形状: {advance_img.shape}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 基础变换（只有resize和normalize）
    img = base_img.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    axes[1].imshow(img)
    axes[1].set_title('Base Transform\n(Resize + Normalize)')
    axes[1].axis('off')

    # 高级变换（包含数据增强）
    img = advance_img.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    axes[2].imshow(img)
    axes[2].set_title('Advanced Transform\n(Crop + Flip + ColorJitter + Grayscale)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/mini_dino_augmentation_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: outputs/mini_dino_augmentation_demo.png")
    plt.close()


def demo_training_augmentation():
    """演示训练脚本中使用的数据增强"""
    print("\n" + "="*60)
    print("方式3: 训练脚本增强（包含 GaussianBlur）")
    print("="*60)

    # 创建训练用的变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0),
                                    interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载测试图像
    image = Image.open('data/koala_test/koala_0.png').convert('RGB')

    # 生成多个增强样本
    augmented_images = [train_transform(image) for _ in range(6)]

    print(f"\n生成了 {len(augmented_images)} 个增强样本")
    print(f"每个样本形状: {augmented_images[0].shape}")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, img_tensor in enumerate(augmented_images):
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f'Augmented Sample {i+1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/training_augmentation_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: outputs/training_augmentation_demo.png")
    plt.close()


def compare_augmentation_strategies():
    """比较不同增强策略的效果"""
    print("\n" + "="*60)
    print("增强策略对比")
    print("="*60)

    strategies = {
        'Base Transform': base_transform(224, 224),
        'Advanced Transform': advance_transform(224, 224),
        'Training Transform': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    # 加载测试图像
    image = Image.open('data/koala_test/koala_0.png').convert('RGB')

    # 应用不同策略
    results = {}
    for name, transform in strategies.items():
        results[name] = transform(image)
        print(f"{name}: {results[name].shape}")

    # 可视化对比
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 不同策略的结果
    for i, (name, img_tensor) in enumerate(results.items(), 1):
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(name)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/augmentation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n对比结果已保存到: outputs/augmentation_comparison.png")
    plt.close()


def analyze_augmentation_parameters():
    """分析不同增强参数的影响"""
    print("\n" + "="*60)
    print("增强参数分析")
    print("="*60)

    print("\n1. ImageNet 归一化参数:")
    print(f"   Mean: [0.485, 0.456, 0.406]")
    print(f"   Std:  [0.229, 0.224, 0.225]")
    print(f"   这些是在 ImageNet 数据集上计算的统计值")

    print("\n2. ColorJitter 参数:")
    print(f"   Brightness: 0.4  (亮度变化 ±40%)")
    print(f"   Contrast:   0.4  (对比度变化 ±40%)")
    print(f"   Saturation: 0.2  (饱和度变化 ±20%)")
    print(f"   Hue:        0.1  (色调变化 ±10%)")

    print("\n3. RandomResizedCrop 参数:")
    print(f"   Full DINOv3 Global: scale=(0.32, 1.0)  覆盖32%-100%的图像")
    print(f"   Full DINOv3 Local:  scale=(0.05, 0.32) 覆盖5%-32%的图像")
    print(f"   Mini-DINO:          scale=(0.2, 1.0)   覆盖20%-100%的图像")
    print(f"   Training Script:    scale=(0.4, 1.0)   覆盖40%-100%的图像")

    print("\n4. GaussianBlur 参数:")
    print(f"   Kernel size: 9 或 23")
    print(f"   Sigma range: (0.1, 2.0)")
    print(f"   应用概率:")
    print(f"     - Global Crop 1: 100% (总是模糊)")
    print(f"     - Global Crop 2: 10%  (偶尔模糊)")
    print(f"     - Local Crops:   50%  (一半概率)")

    print("\n5. 其他增强:")
    print(f"   RandomGrayscale:     p=0.2  (20%概率转灰度)")
    print(f"   RandomHorizontalFlip: p=0.5  (50%概率水平翻转)")
    print(f"   RandomSolarize:      p=0.2  (20%概率反色，仅Global Crop 2)")


if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)

    print("\n" + "="*60)
    print("DINOv3 图像处理完整演示")
    print("="*60)

    # 1. 完整 DINOv3 Multi-Crop 增强
    try:
        demo_full_dinov3_augmentation()
    except Exception as e:
        print(f"完整 DINOv3 演示失败: {e}")

    # 2. 简化 Mini-DINO 增强
    try:
        demo_mini_dino_augmentation()
    except Exception as e:
        print(f"Mini-DINO 演示失败: {e}")

    # 3. 训练脚本增强
    try:
        demo_training_augmentation()
    except Exception as e:
        print(f"训练增强演示失败: {e}")

    # 4. 策略对比
    try:
        compare_augmentation_strategies()
    except Exception as e:
        print(f"策略对比失败: {e}")

    # 5. 参数分析
    analyze_augmentation_parameters()

    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
