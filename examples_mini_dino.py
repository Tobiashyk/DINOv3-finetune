"""
Mini-DINOv3 使用示例

这个脚本展示了如何使用 Mini-DINOv3 进行特征提取和可视化。
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model.ssl_ref import MiniDINO
from src.data.test_datamodule import load_images
from src.utils.image_process import make_transform
from src.utils.visualize import pca_transform_features


def extract_features_example():
    """示例：使用 Mini-DINOv3 提取图像特征"""
    print("\n" + "="*60)
    print("示例 1: 特征提取")
    print("="*60)

    # 1. 创建模型（使用预训练的 DINOv3 backbone）
    print("\n加载模型...")
    model = MiniDINO(
        backbone_name='dinov3_vits14',  # ViT-Small, 384维特征
        out_dim=8192,                    # 8192个原型
    )
    model.eval()

    # 2. 准备图像
    print("准备测试图像...")
    # 假设你有一些图像
    images = torch.randn(4, 3, 224, 224)  # 4张图像

    # 3. 提取特征
    print("提取特征...")
    with torch.no_grad():
        # 方法1: 提取 CLS token 特征（全局图像表示）
        cls_features = model.student_backbone(images)['x_norm_clstoken']
        print(f"✓ CLS token 特征形状: {cls_features.shape}")  # [4, 384]

        # 方法2: 提取 patch token 特征（局部特征）
        patch_features = model.student_backbone(images)['x_norm_patchtokens']
        print(f"✓ Patch token 特征形状: {patch_features.shape}")  # [4, 256, 384]

        # 方法3: 提取原型空间的表示
        prototype_logits = model.student_head(cls_features)
        print(f"✓ 原型 logits 形状: {prototype_logits.shape}")  # [4, 8192]

    print("\n特征提取完成！")
    return cls_features, patch_features


def training_step_example():
    """示例：单步训练"""
    print("\n" + "="*60)
    print("示例 2: 训练步骤")
    print("="*60)

    # 1. 创建模型
    print("\n初始化模型...")
    model = MiniDINO(backbone_name='dinov3_vits14', out_dim=8192)

    # 2. 创建优化器
    optimizer = torch.optim.AdamW(
        model.get_student_parameters(),
        lr=0.001,
        weight_decay=0.04,
    )

    # 3. 准备一个 batch
    images = torch.randn(4, 3, 224, 224)

    # 4. 训练步骤
    print("\n执行训练步骤...")

    # Forward pass
    outputs = model(images)
    loss = outputs['loss']
    print(f"✓ 损失值: {loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.get_student_parameters(),
        max_norm=3.0
    )
    print(f"✓ 梯度范数: {grad_norm:.4f}")

    # Optimizer step
    optimizer.step()

    # Update teacher with EMA
    model.update_teacher(momentum=0.996)
    print("✓ 教师网络已更新（EMA）")

    print("\n训练步骤完成！")


def visualize_features_example():
    """示例：特征可视化"""
    print("\n" + "="*60)
    print("示例 3: 特征可视化")
    print("="*60)

    # 1. 加载模型
    print("\n加载模型...")
    model = MiniDINO(backbone_name='dinov3_vits14', out_dim=8192)
    model.eval()

    # 2. 加载图像
    print("加载图像...")
    # 假设你有一些图像
    images = torch.randn(2, 3, 224, 224)

    # 3. 提取 patch 特征
    print("提取 patch 特征...")
    with torch.no_grad():
        patch_features = model.student_backbone(images)['x_norm_patchtokens']
        # patch_features: [2, 256, 384] (2张图，256个patch，384维特征)

    # 4. PCA 可视化
    print("应用 PCA 降维...")
    patch_h, patch_w = 16, 16  # 假设是 16x16 的 patch 网格
    pca_images = pca_transform_features(patch_features, patch_h, patch_w, n_components=3)
    print(f"✓ PCA 特征形状: {pca_images.shape}")  # [2, 16, 16, 3]

    # 5. 可视化
    print("保存可视化结果...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(2):
        axes[i].imshow(pca_images[i])
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1} - PCA Features')

    output_path = 'outputs/feature_visualization.png'
    Path('outputs').mkdir(exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"✓ 可视化已保存到: {output_path}")

    print("\n特征可视化完成！")


def compare_student_teacher_example():
    """示例：比较学生和教师的输出"""
    print("\n" + "="*60)
    print("示例 4: 学生 vs 教师")
    print("="*60)

    # 1. 创建模型
    print("\n初始化模型...")
    model = MiniDINO(backbone_name='dinov3_vits14', out_dim=8192)
    model.eval()

    # 2. 准备图像
    images = torch.randn(4, 3, 224, 224)

    # 3. 比较学生和教师
    print("\n比较学生和教师的输出...")
    with torch.no_grad():
        # 学生输出
        student_features = model.student_backbone(images)['x_norm_clstoken']
        student_logits = model.student_head(student_features)

        # 教师输出
        teacher_features = model.teacher_backbone(images)['x_norm_clstoken']
        teacher_logits = model.teacher_head(teacher_features)

        # 计算差异
        feature_diff = torch.norm(student_features - teacher_features, dim=1).mean()
        logit_diff = torch.norm(student_logits - teacher_logits, dim=1).mean()

        print(f"✓ 特征差异（L2范数）: {feature_diff.item():.4f}")
        print(f"✓ Logits 差异（L2范数）: {logit_diff.item():.4f}")

        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(
            student_features, teacher_features, dim=1
        ).mean()
        print(f"✓ 余弦相似度: {cos_sim.item():.4f}")

    print("\n比较完成！")
    print("注意：训练开始时，学生和教师应该非常相似（因为教师是学生的副本）")
    print("随着训练进行，它们会略有不同，但教师会通过 EMA 跟随学生")


def loss_components_example():
    """示例：理解损失的各个组成部分"""
    print("\n" + "="*60)
    print("示例 5: 损失组成部分")
    print("="*60)

    from src.model.loss import DINOLoss

    # 1. 创建损失函数
    print("\n创建 DINO 损失...")
    loss_fn = DINOLoss(out_dim=8192, student_temp=0.1, center_momentum=0.9)

    # 2. 创建假数据
    batch_size = 4
    student_logits = torch.randn(batch_size, 8192)
    teacher_logits = torch.randn(batch_size, 8192)

    # 3. 应用 Sinkhorn-Knopp
    print("\n应用 Sinkhorn-Knopp 算法...")
    teacher_probs = loss_fn.sinkhorn_knopp_teacher(
        teacher_logits,
        teacher_temp=0.04,
        n_iterations=3
    )

    # 检查概率分布
    prob_sums = teacher_probs.sum(dim=1)
    print(f"✓ 教师概率和: {prob_sums.tolist()}")
    print(f"  （应该都接近 1.0）")

    # 检查熵
    entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=1).mean()
    print(f"✓ 教师分布的熵: {entropy.item():.4f}")
    print(f"  （熵越高，分布越均匀）")

    # 4. 计算损失
    print("\n计算 DINO 损失...")
    loss = loss_fn(student_logits, teacher_probs)
    print(f"✓ 损失值: {loss.item():.4f}")

    # 5. 更新中心
    print("\n更新中心...")
    old_center = loss_fn.center.clone()
    loss_fn.update_center(teacher_logits)
    center_change = torch.norm(loss_fn.center - old_center)
    print(f"✓ 中心变化（L2范数）: {center_change.item():.4f}")

    print("\n损失分析完成！")


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("Mini-DINOv3 使用示例")
    print("="*70)

    print("\n这些示例展示了如何使用 Mini-DINOv3 的各个组件。")
    print("注意：某些示例需要从 torch.hub 下载预训练模型（需要网络连接）。")

    try:
        # 示例 1: 特征提取
        # extract_features_example()

        # 示例 2: 训练步骤
        # training_step_example()

        # 示例 3: 特征可视化
        # visualize_features_example()

        # 示例 4: 学生 vs 教师
        # compare_student_teacher_example()

        # 示例 5: 损失组成部分（不需要网络）
        loss_components_example()

        print("\n" + "="*70)
        print("✓ 所有示例运行成功！")
        print("="*70)

        print("\n提示：")
        print("- 取消注释上面的函数调用来运行其他示例")
        print("- 某些示例需要网络连接来下载预训练模型")
        print("- 你可以修改这些示例来适应你的需求")

    except Exception as e:
        print("\n" + "="*70)
        print("✗ 示例运行失败")
        print("="*70)
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
