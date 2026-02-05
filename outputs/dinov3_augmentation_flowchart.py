"""
DINOv3 DataAugmentationDINO Flowchart
Complete data augmentation pipeline visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 30)
ax.axis('off')

# Color scheme
color_input = '#E8F4F8'
color_geometric = '#FFE5CC'
color_color = '#FFD6E8'
color_blur = '#D4E8D4'
color_normalize = '#E8E8FF'
color_output = '#FFE6CC'

def draw_box(ax, x, y, width, height, text, color, fontsize=10, bold=False):
    """Draw rounded rectangle box"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize, weight=weight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """Draw arrow"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color='black',
        linewidth=2,
        mutation_scale=20
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def draw_dashed_box(ax, x, y, width, height, label, color):
    """Draw dashed box for optional flows"""
    rect = mpatches.Rectangle(
        (x, y), width, height,
        linewidth=2,
        edgecolor=color,
        facecolor='none',
        linestyle='--'
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height + 0.3, label,
            ha='center', va='bottom', fontsize=11, weight='bold', color=color)

# ============= Title =============
ax.text(11, 29, 'DINOv3 DataAugmentationDINO Pipeline',
        ha='center', va='top', fontsize=20, weight='bold')
ax.text(11, 28, 'Multi-Crop Data Augmentation Strategy',
        ha='center', va='top', fontsize=13, style='italic', color='gray')

# ============= Input Image =============
draw_box(ax, 9.5, 26, 3, 1, 'Input Image\nPIL Image (RGB)', color_input, fontsize=11, bold=True)

# ============= Optional: Shared Color Jitter =============
draw_dashed_box(ax, 8, 24, 6, 1.5, 'Optional: share_color_jitter=True', '#FF6B6B')
draw_box(ax, 8.5, 24.3, 5, 0.8, 'ColorJitter (80%) + Grayscale (20%)', color_color, fontsize=9)
draw_arrow(ax, 11, 26, 11, 25.1, 'if shared')

# ============= Branch Point =============
draw_arrow(ax, 11, 24, 11, 23)
ax.text(11, 23.3, 'Branch Processing', ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))

# ============= Left Branch: Global Crop 1 =============
left_x = 2

# Geometric augmentation
draw_box(ax, left_x, 21, 3.5, 1, 'Geometric Aug\nRandomResizedCrop\nscale=(0.32, 1.0)\nFlip (50%)',
         color_geometric, fontsize=9)
draw_arrow(ax, 11, 23, left_x + 1.75, 22)
ax.text(left_x + 1.75, 22.5, 'im1_base', ha='center', fontsize=8, style='italic', color='blue')

# Color jitter (if not shared)
draw_box(ax, left_x, 19.5, 3.5, 0.8, 'ColorJitter (80%)\nif not shared',
         color_color, fontsize=8)
draw_arrow(ax, left_x + 1.75, 21, left_x + 1.75, 20.3)

# Gaussian blur
draw_box(ax, left_x, 18, 3.5, 0.8, 'GaussianBlur (100%)\nAlways blur!',
         color_blur, fontsize=9, bold=True)
draw_arrow(ax, left_x + 1.75, 19.5, left_x + 1.75, 18.8)

# Normalize
draw_box(ax, left_x, 16.5, 3.5, 0.8, 'Normalize\nImageNet mean/std',
         color_normalize, fontsize=9)
draw_arrow(ax, left_x + 1.75, 18, left_x + 1.75, 17.3)

# Output
draw_box(ax, left_x, 15, 3.5, 0.8, 'Global Crop 1\n[3, 224, 224]',
         color_output, fontsize=9, bold=True)
draw_arrow(ax, left_x + 1.75, 16.5, left_x + 1.75, 15.8)

# ============= Middle Branch: Global Crop 2 =============
mid_x = 9.25

# Geometric augmentation
draw_box(ax, mid_x, 21, 3.5, 1, 'Geometric Aug\nRandomResizedCrop\nscale=(0.32, 1.0)\nFlip (50%)',
         color_geometric, fontsize=9)
draw_arrow(ax, 11, 23, mid_x + 1.75, 22)
ax.text(mid_x + 1.75, 22.5, 'im2_base', ha='center', fontsize=8, style='italic', color='blue')

# Color jitter (if not shared)
draw_box(ax, mid_x, 19.5, 3.5, 0.8, 'ColorJitter (80%)\nif not shared',
         color_color, fontsize=8)
draw_arrow(ax, mid_x + 1.75, 21, mid_x + 1.75, 20.3)

# Gaussian blur + Solarize
draw_box(ax, mid_x, 18, 3.5, 0.8, 'GaussianBlur (10%)\nSolarize (20%)',
         color_blur, fontsize=9)
draw_arrow(ax, mid_x + 1.75, 19.5, mid_x + 1.75, 18.8)

# Normalize
draw_box(ax, mid_x, 16.5, 3.5, 0.8, 'Normalize\nImageNet mean/std',
         color_normalize, fontsize=9)
draw_arrow(ax, mid_x + 1.75, 18, mid_x + 1.75, 17.3)

# Output
draw_box(ax, mid_x, 15, 3.5, 0.8, 'Global Crop 2\n[3, 224, 224]',
         color_output, fontsize=9, bold=True)
draw_arrow(ax, mid_x + 1.75, 16.5, mid_x + 1.75, 15.8)

# ============= Right Branch: Local Crops (8x) =============
right_x = 16.5

# Geometric augmentation
draw_box(ax, right_x, 21, 3.5, 1, 'Geometric Aug\nRandomResizedCrop\nscale=(0.05, 0.32)\nFlip (50%)',
         color_geometric, fontsize=9)
draw_arrow(ax, 11, 23, right_x + 1.75, 22)
ax.text(right_x + 1.75, 22.5, 'x8 times', ha='center', fontsize=8, style='italic', color='red', weight='bold')

# Color jitter (if not shared)
draw_box(ax, right_x, 19.5, 3.5, 0.8, 'ColorJitter (80%)\nif not shared',
         color_color, fontsize=8)
draw_arrow(ax, right_x + 1.75, 21, right_x + 1.75, 20.3)

# Gaussian blur
draw_box(ax, right_x, 18, 3.5, 0.8, 'GaussianBlur (50%)',
         color_blur, fontsize=9)
draw_arrow(ax, right_x + 1.75, 19.5, right_x + 1.75, 18.8)

# Normalize
draw_box(ax, right_x, 16.5, 3.5, 0.8, 'Normalize\nImageNet mean/std',
         color_normalize, fontsize=9)
draw_arrow(ax, right_x + 1.75, 18, right_x + 1.75, 17.3)

# Output
draw_box(ax, right_x, 15, 3.5, 0.8, 'Local Crops x8\n[3, 96, 96]',
         color_output, fontsize=9, bold=True)
draw_arrow(ax, right_x + 1.75, 16.5, right_x + 1.75, 15.8)

# ============= Optional Branch: Teacher Crops =============
draw_dashed_box(ax, 0.5, 12, 6.5, 2, 'Optional: teacher_no_color_jitter=True', '#9B59B6')
draw_box(ax, 1, 13, 2.5, 0.8, 'im1_base\n-> Normalize', color_normalize, fontsize=8)
draw_box(ax, 4, 13, 2.5, 0.8, 'im2_base\n-> Normalize', color_normalize, fontsize=8)
draw_arrow(ax, left_x + 1.75, 21, 2.25, 13.8, '', '->')
draw_arrow(ax, mid_x + 1.75, 21, 5.25, 13.8, '', '->')
draw_box(ax, 1, 12.3, 5.5, 0.5, 'global_crops_teacher [2]', color_output, fontsize=8)

# ============= Optional Branch: Gram Teacher Crops =============
draw_dashed_box(ax, 7.5, 12, 6.5, 2, 'Optional: gram_teacher_crops_size != None', '#E67E22')
draw_box(ax, 8, 13, 2.5, 0.8, 'Resize + Normalize\nif no_distortions', color_normalize, fontsize=8)
draw_box(ax, 11, 13, 2.5, 0.8, 'Resize + Normalize\nif no_distortions', color_normalize, fontsize=8)
draw_arrow(ax, left_x + 1.75, 21, 9.25, 13.8, '', '->')
draw_arrow(ax, mid_x + 1.75, 21, 12.25, 13.8, '', '->')
draw_box(ax, 8, 12.3, 5.5, 0.5, 'gram_teacher_crops [2]', color_output, fontsize=8)

# ============= Optional Branch: Local Crops Subset =============
draw_dashed_box(ax, 14.5, 12, 7, 2, 'Optional: local_crops_subset_of_global_crops=True', '#16A085')
draw_box(ax, 15, 13, 6, 0.8, 'Extract from im1_base/im2_base\nAligned to patch_size boundary', color_geometric, fontsize=8)
draw_arrow(ax, left_x + 1.75, 21, 18, 13.8, '', '->')
draw_arrow(ax, mid_x + 1.75, 21, 18, 13.8, '', '->')
draw_box(ax, 15, 12.3, 6, 0.5, 'local_crops [8] + offsets', color_output, fontsize=8)

# ============= Final Output Dictionary =============
draw_box(ax, 7, 9, 8, 2,
         'output = {\n'
         '  "global_crops": [crop1, crop2],\n'
         '  "global_crops_teacher": [...],\n'
         '  "local_crops": [crop1, ..., crop8],\n'
         '  "gram_teacher_crops": [...],  # optional\n'
         '  "offsets": [...]  # optional\n'
         '}',
         '#FFF9C4', fontsize=9, bold=True)

# Converging arrows
draw_arrow(ax, left_x + 1.75, 15, 9, 11)
draw_arrow(ax, mid_x + 1.75, 15, 11, 11)
draw_arrow(ax, right_x + 1.75, 15, 13, 11)

# ============= Legend =============
legend_y = 7
ax.text(11, legend_y + 0.5, 'Legend', ha='center', fontsize=12, weight='bold')

legend_items = [
    (color_input, 'Input'),
    (color_geometric, 'Geometric'),
    (color_color, 'Color Distortion'),
    (color_blur, 'Blur/Solarize'),
    (color_normalize, 'Normalize'),
    (color_output, 'Output'),
]

for i, (color, label) in enumerate(legend_items):
    x = 3.5 + (i % 3) * 5
    y = legend_y - 0.5 - (i // 3) * 0.8
    draw_box(ax, x, y, 2, 0.5, label, color, fontsize=9)

# ============= Key Parameters Comparison =============
param_y = 4.5
ax.text(11, param_y + 0.5, 'Key Parameters Comparison', ha='center', fontsize=12, weight='bold')

params_text = """
+------------------+--------------+--------------+--------------+
|                  | Global Crop 1| Global Crop 2| Local Crops  |
+------------------+--------------+--------------+--------------+
| Crop Scale       | 32%-100%     | 32%-100%     | 5%-32%       |
| Output Size      | 224x224      | 224x224      | 96x96        |
| GaussianBlur     | 100% *       | 10%          | 50%          |
| Solarize         | 0%           | 20% *        | 0%           |
| Number           | 1            | 1            | 8            |
+------------------+--------------+--------------+--------------+
"""

ax.text(11, param_y - 0.5, params_text, ha='center', va='top', fontsize=8,
        family='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0'))

# ============= Design Philosophy =============
design_y = 1
ax.text(11, design_y,
        'Core Design: Asymmetric Augmentation - Global Crop 1 (strong aug) prevents shortcut learning,\n'
        'Global Crop 2 (weak aug) provides clear view, Local Crops (medium aug) learn local features',
        ha='center', fontsize=9, style='italic', weight='bold', color='#D32F2F',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE'))

plt.tight_layout()
plt.savefig('/home/lcy/Documents/zju/hyk/DINOv3-finetune/outputs/dinov3_augmentation_flowchart.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Flowchart saved to: outputs/dinov3_augmentation_flowchart.png")
plt.close()

print("\nFlowchart generation complete!")
print("=" * 60)
print("Main Pipeline:")
print("1. Input original image")
print("2. Optional: Shared color jitter (share_color_jitter=True)")
print("3. Branch processing:")
print("   - Left: Global Crop 1 (100% GaussianBlur)")
print("   - Middle: Global Crop 2 (10% GaussianBlur + 20% Solarize)")
print("   - Right: Local Crops x8 (50% GaussianBlur)")
print("4. Optional branches:")
print("   - Teacher Crops (no color jitter)")
print("   - Gram Teacher Crops (optional no distortions)")
print("   - Local Crops Subset (extract from global crops)")
print("5. Output dictionary contains all crop results")
print("=" * 60)
