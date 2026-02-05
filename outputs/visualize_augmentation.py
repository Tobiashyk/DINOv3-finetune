"""
DINOv3 Data Augmentation Visual Comparison
Visualize the actual augmentation effects for different crop types
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dinov3'))

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from dinov3.data.augmentations import DataAugmentationDINO
    DINOV3_AVAILABLE = True
except ImportError:
    print("Warning: Cannot import DINOv3 module")
    DINOV3_AVAILABLE = False

def denormalize(tensor):
    """Denormalize tensor for visualization"""
    img = tensor.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)

def visualize_augmentation_comparison():
    """Create comprehensive visualization of augmentation effects"""

    if not DINOV3_AVAILABLE:
        print("DINOv3 not available, skipping visualization")
        return

    # Load test image
    image_path = 'data/koala_test/koala_0.png'
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        return

    image = Image.open(image_path).convert('RGB')

    # Create augmentation
    augmentation = DataAugmentationDINO(
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96,
    )

    # Apply augmentation
    output = augmentation(image)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle('DINOv3 Data Augmentation - Visual Comparison',
                 fontsize=20, weight='bold', y=0.98)

    # ========== Original Image ==========
    ax_orig = fig.add_subplot(gs[0, :2])
    ax_orig.imshow(image)
    ax_orig.set_title('Original Image\n(Input)', fontsize=14, weight='bold')
    ax_orig.axis('off')

    # Add annotation box
    textstr = 'Input: PIL Image (RGB)\nSize: Variable'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax_orig.text(0.5, -0.15, textstr, transform=ax_orig.transAxes,
                fontsize=10, verticalalignment='top', ha='center', bbox=props)

    # ========== Global Crop 1 (Strong Augmentation) ==========
    ax_g1 = fig.add_subplot(gs[0, 2:4])
    ax_g1.imshow(denormalize(output['global_crops'][0]))
    ax_g1.set_title('Global Crop 1 - Strong Aug\n(100% Blur)',
                    fontsize=14, weight='bold', color='red')
    ax_g1.axis('off')

    # Add red border
    rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax_g1.transAxes,
                              fill=False, edgecolor='red', linewidth=4)
    ax_g1.add_patch(rect)

    textstr = 'Steps:\n1. RandomResizedCrop (32-100%)\n2. ColorJitter (80%)\n3. GaussianBlur (100%) ⭐\n4. Normalize\nOutput: [3, 224, 224]'
    props = dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8)
    ax_g1.text(0.5, -0.15, textstr, transform=ax_g1.transAxes,
              fontsize=9, verticalalignment='top', ha='center', bbox=props)

    # ========== Global Crop 2 (Weak Augmentation) ==========
    ax_g2 = fig.add_subplot(gs[0, 4:6])
    ax_g2.imshow(denormalize(output['global_crops'][1]))
    ax_g2.set_title('Global Crop 2 - Weak Aug\n(10% Blur, 20% Solarize)',
                    fontsize=14, weight='bold', color='orange')
    ax_g2.axis('off')

    # Add orange border
    rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax_g2.transAxes,
                              fill=False, edgecolor='orange', linewidth=4)
    ax_g2.add_patch(rect)

    textstr = 'Steps:\n1. RandomResizedCrop (32-100%)\n2. ColorJitter (80%)\n3. GaussianBlur (10%) + Solarize (20%) ⭐\n4. Normalize\nOutput: [3, 224, 224]'
    props = dict(boxstyle='round', facecolor='#ffe6cc', alpha=0.8)
    ax_g2.text(0.5, -0.15, textstr, transform=ax_g2.transAxes,
              fontsize=9, verticalalignment='top', ha='center', bbox=props)

    # ========== Local Crops (Medium Augmentation) ==========
    for i in range(8):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col:col+1])
        ax.imshow(denormalize(output['local_crops'][i]))
        ax.set_title(f'Local Crop {i+1}', fontsize=11, weight='bold', color='green')
        ax.axis('off')

        # Add green border
        rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                  fill=False, edgecolor='green', linewidth=3)
        ax.add_patch(rect)

    # Add local crops description
    ax_local_desc = fig.add_subplot(gs[1:3, 4:6])
    ax_local_desc.axis('off')

    desc_text = '''Local Crops - Medium Augmentation

Processing Steps:
1. RandomResizedCrop (5-32%)
   → Much smaller crop region

2. ColorJitter (80%)
   → Same as global crops

3. GaussianBlur (50%)
   → Medium blur probability

4. Normalize
   → ImageNet mean/std

Output: [3, 96, 96] × 8

Purpose:
• Learn local features
• Focus on fine-grained details
• Complement global crops
• Increase training diversity
'''

    props = dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8)
    ax_local_desc.text(0.5, 0.5, desc_text, transform=ax_local_desc.transAxes,
                      fontsize=10, verticalalignment='center', ha='center',
                      bbox=props, family='monospace')

    # ========== Summary Table ==========
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')

    summary_text = '''
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                              AUGMENTATION PROBABILITY COMPARISON                                                      ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Operation              │  Global Crop 1 (Strong)  │  Global Crop 2 (Weak)   │  Local Crops (Medium)   │  Purpose                    ║
╠═════════════════════════╪══════════════════════════╪═════════════════════════╪═════════════════════════╪═════════════════════════════╣
║  RandomResizedCrop      │  100% (32-100%)          │  100% (32-100%)         │  100% (5-32%)           │  Spatial augmentation       ║
║  RandomHorizontalFlip   │  50%                     │  50%                    │  50%                    │  Spatial augmentation       ║
║  ColorJitter            │  80%                     │  80%                    │  80%                    │  Color augmentation         ║
║  RandomGrayscale        │  20%                     │  20%                    │  20%                    │  Color augmentation         ║
║  GaussianBlur           │  100% ⭐ (Always!)       │  10%                    │  50%                    │  Prevent shortcut learning  ║
║  RandomSolarize         │  0%                      │  20% ⭐                 │  0%                     │  Additional distortion      ║
╠═════════════════════════╪══════════════════════════╪═════════════════════════╪═════════════════════════╪═════════════════════════════╣
║  Output Shape           │  [3, 224, 224]           │  [3, 224, 224]          │  [3, 96, 96]            │                             ║
║  Count                  │  1                       │  1                      │  8                      │  Total: 10 images           ║
║  Augmentation Strength  │  🔴 Strong               │  🟡 Weak                │  🟠 Medium              │                             ║
╚═════════════════════════╧══════════════════════════╧═════════════════════════╧═════════════════════════╧═════════════════════════════╝

Key Design Principle: ASYMMETRIC AUGMENTATION
• Global Crop 1 (100% blur) forces the model to learn robust features that work even with heavy distortions
• Global Crop 2 (weak aug) provides clear view for learning fine-grained details
• Local Crops (medium aug) balance between robustness and detail, focus on local features
• This strategy prevents the model from learning simple shortcuts (e.g., color histogram matching)
'''

    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, verticalalignment='center', ha='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Save figure
    output_path = '/home/lcy/Documents/zju/hyk/DINOv3-finetune/outputs/augmentation_visual_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {output_path}")
    plt.close()

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)
    print("\nGenerated images:")
    print(f"  • 2 Global Crops: [3, 224, 224]")
    print(f"  • 8 Local Crops:  [3, 96, 96]")
    print(f"  • Total: 10 different views of the same image")
    print("\nAugmentation strengths:")
    print(f"  🔴 Global Crop 1: Strong (100% blur)")
    print(f"  🟡 Global Crop 2: Weak (10% blur, 20% solarize)")
    print(f"  🟠 Local Crops:   Medium (50% blur)")
    print("="*80)

if __name__ == '__main__':
    visualize_augmentation_comparison()
