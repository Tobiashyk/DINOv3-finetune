"""
从四种材料的clean文件夹中分别随机提取40张图片，
复制到train_pic文件夹下的对应材料文件夹中，并添加_train_try后缀。
每次运行会从每种材料中随机选择不同的图片组合。
"""

import os
import shutil
import random
from pathlib import Path


def extract_train_try_samples(
    source_base_dir: str = "data/train_pic",
    target_base_dir: str = "data/train_pic",
    num_samples: int = 40,
    random_seed: int = None
):
    """
    从每种材料的images_clean文件夹中随机提取N张图片，
    复制到目标目录并添加_train_try后缀。
    每次运行会随机选择不同的图片组合。

    Args:
        source_base_dir: 源数据目录路径
        target_base_dir: 目标输出目录路径
        num_samples: 每种材料提取的图片数量
        random_seed: 随机种子，设置后可复现结果（可选）
    """
    # 设置随机种子（如果提供）
    if random_seed is not None:
        random.seed(random_seed)
    # 四种材料文件夹
    materials = ["MoS2_1T_train", "MoS2_2H_train", "graphene_AA_train", "graphene_AB_train"]

    # 支持的图片扩展名
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}

    for material in materials:
        # 源clean文件夹路径
        clean_dir = Path(source_base_dir) / material / "images_clean"

        # 目标文件夹路径
        target_dir = Path(target_base_dir) / 'train_try_test5' / material

        # 检查源文件夹是否存在
        if not clean_dir.exists():
            print(f"警告: 源文件夹不存在 - {clean_dir}")
            continue

        # 创建目标文件夹
        target_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图片文件
        image_files = [
            f for f in clean_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        # 检查图片数量是否足够
        if len(image_files) < num_samples:
            print(f"警告: {material} 只有 {len(image_files)} 张图片，少于需要的 {num_samples} 张")
            selected_files = image_files
        else:
            # 随机选择N张图片（不放回抽样）
            selected_files = random.sample(image_files, num_samples)

        # 复制并重命名图片
        copied_count = 0
        for img_path in selected_files:
            # 构建新文件名: 原文件名_train_try.扩展名
            new_filename = f"{img_path.stem}_train_try{img_path.suffix}"
            target_path = target_dir / new_filename

            try:
                shutil.copy2(img_path, target_path)
                copied_count += 1
            except Exception as e:
                print(f"错误: 复制文件失败 {img_path} -> {target_path}: {e}")

        print(f"{material}: 成功复制 {copied_count} 张图片到 {target_dir}")


def main():
    """主函数"""
    print("开始提取训练样本...")
    extract_train_try_samples()
    print("\n提取完成！")


if __name__ == "__main__":
    main()
