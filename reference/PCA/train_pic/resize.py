import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # 如果没有安装 tqdm，可以运行 pip install tqdm，或者把相关进度条代码删掉

def resize_images_like_training(source_dir, dest_dir, target_size=1024):

    # 1. 确保目标文件夹存在
    os.makedirs(dest_dir, exist_ok=True)

    # 2. 定义转换方法
    # 对应你训练代码 main 函数中的: transforms.Resize((1024,1024))
    # PyTorch 默认使用 InterpolationMode.BILINEAR (双线性插值)
    transform = transforms.Resize((target_size, target_size))

    # 3. 获取所有图片文件
    source_path = Path(source_dir)
    # 支持常见的图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    image_files = []
    for ext in extensions:
        # 这里加上 sorted 保持和你训练代码一样的严谨性
        image_files.extend(sorted(list(source_path.glob(ext))))

    print(f"找到 {len(image_files)} 张图片，准备处理...")

    # 4. 循环处理
    for img_path in tqdm(image_files, desc="Resizing"):
        try:
            # A. 读取方式与训练代码 Dataset.__getitem__ 保持一致
            # img = Image.open(img_path).convert('RGB')
            img = Image.open(img_path).convert('RGB')

            # B. 应用缩放
            # 这里输入是 PIL Image，输出也是 PIL Image (因为没有加 ToTensor)
            resized_img = transform(img)

            # C. 保存图片
            save_path = Path(dest_dir) / img_path.name
            
            # 如果是 png 格式建议用 optimize=True 减小体积，不影响像素值
            resized_img.save(save_path, quality=100)
            
        except Exception as e:
            print(f"处理图片 {img_path.name} 时出错: {e}")

    print(f"处理完成！所有图片已保存至: {dest_dir}")
    print(f"分辨率已统一为: {target_size}x{target_size}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 输入文件夹 (256分辨率图片的文件夹)
    source_folder = "/home/abc/projects/DINOv3/PCA/train_pic/Sim_2H_256" 
    
    # 输出文件夹 (保存1024分辨率图片的文件夹)
    target_folder = "/home/abc/projects/DINOv3/PCA/train_pic/Sim_2H_1024"
    
    # ===========================================

    resize_images_like_training(source_folder, target_folder)