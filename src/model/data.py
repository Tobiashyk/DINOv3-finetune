from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
import os

from src.utils.image_process import global_transform, local_transform, _get_color_jittering


# class SimpleImageDataset(Dataset):
#     def __init__(self, image_dir: str, transforms=None):
#         self.image_dir = image_dir
#         self.transforms = transforms
#         self.image_paths = [
#             os.path.join(image_dir, fname)
#             for fname in os.listdir(image_dir)
#             if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
#         ]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):

#         img_path = self.image_paths[idx]
#         original_image = Image.open(img_path).convert('RGB')

#         images = []
#         for transform in self.transforms:
#             transformed_image = transform(original_image)
#             images.append(transformed_image)
#         return images
class SimpleImageDataset(Dataset):
    def __init__(self, image_dir: str, global_transform=None, local_transform=None,
                 color_jitter=True, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8,
                 image_size: int = 512, num_global_crops: int = 2, num_local_crops: int = 8):
        self.image_paths = []
        for root_dir in image_dir:
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                paths = sorted(list(Path(root_dir).glob(ext)))
                self.image_paths.extend(paths)

        self.global_transform = global_transform
        self.local_transform = local_transform
        self.color_jitter = color_jitter
        self.image_size = image_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        if self.color_jitter:
            self.color_jitter_transform = _get_color_jittering(brightness, contrast, saturation, hue, p)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        original_image = Image.open(img_path).convert('RGB')
        original_image = original_image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # 应用颜色抖动（随机亮暗处理）
        if self.color_jitter:
            original_image = self.color_jitter_transform(original_image)

        # 生成多个 global crops（Teacher 和 Student 都会使用）
        global_crops = []
        for _ in range(self.num_global_crops):
            global_crops.append(self.global_transform(original_image))

        # 生成多个 local crops（仅 Student 使用）
        local_crops = []
        for _ in range(self.num_local_crops):
            local_crops.append(self.local_transform(original_image))

        return global_crops, local_crops


class PairedSTEMDataset(Dataset):
    """
    严格配对的STEM图像数据集，用于第二阶段非对称掩码去噪特征蒸馏

    支持多文件夹输入，同时读取干净图和加噪图，确保文件名一一对应，像素级对齐
    """
    def __init__(self, clean_dirs, noisy_dirs, transform=None, image_size: int = 1024):
        """
        Args:
            clean_dirs: 干净图像文件夹路径（str 或 list[str]）
            noisy_dirs: 加噪图像文件夹路径（str 或 list[str]）
            transform: 图像变换（只需要resize到image_size x image_size，不做裁剪）
            image_size: 图像尺寸（默认为1024）
        """
        # 统一转换为列表
        if isinstance(clean_dirs, str):
            clean_dirs = [clean_dirs]
        if isinstance(noisy_dirs, str):
            noisy_dirs = [noisy_dirs]

        if len(clean_dirs) != len(noisy_dirs):
            raise ValueError(f"clean_dirs ({len(clean_dirs)}) 和 noisy_dirs ({len(noisy_dirs)}) 数量必须相同")

        self.clean_dirs = [Path(d) for d in clean_dirs]
        self.noisy_dirs = [Path(d) for d in noisy_dirs]
        self.transform = transform
        self.image_size = image_size

        # 收集所有配对的图像路径
        self.paired_images = []  # [(clean_path, noisy_path), ...]

        for clean_dir, noisy_dir in zip(self.clean_dirs, self.noisy_dirs):
            # 获取当前clean文件夹中的图像
            clean_files = set()
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                clean_files.update(p.name for p in clean_dir.glob(ext))

            # 验证noisy文件夹中是否存在对应文件
            missing_files = []
            for name in sorted(clean_files):
                noisy_path = noisy_dir / name
                if noisy_path.exists():
                    self.paired_images.append((clean_dir / name, noisy_path))
                else:
                    missing_files.append(name)

            if missing_files:
                print(f"警告: {clean_dir} 中有 {len(missing_files)} 个文件在 {noisy_dir} 中不存在，已跳过")

        print(f"PairedSTEMDataset: 共找到 {len(self.paired_images)} 对严格配对的图像")

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.paired_images[idx]

        # 读取配对的干净图和加噪图
        clean_img = Image.open(clean_path).convert('RGB')
        noisy_img = Image.open(noisy_path).convert('RGB')

        # Resize到image_size x image_size（如果需要）
        clean_img = clean_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        noisy_img = noisy_img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # 应用相同的变换
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)

        return clean_img, noisy_img


def gen_dataloader(data_cfg):
    global_crop_size = getattr(data_cfg, 'global_crop_size', 512)
    local_crop_size = getattr(data_cfg, 'local_crop_size', 256)

    if global_crop_size % 16 != 0:
        raise ValueError(f"global_crop_size ({global_crop_size}) 必须是 16 的倍数 (patch_size=16)")
    if local_crop_size % 16 != 0:
        raise ValueError(f"local_crop_size ({local_crop_size}) 必须是 16 的倍数 (patch_size=16)")

    g_transform = global_transform(image_size=global_crop_size)
    l_transform = local_transform(
        local_crop_size=local_crop_size,
        local_crop_scale=tuple(data_cfg.local_crop_scale),
        horizontal_flips=data_cfg.horizontal_flips,
    )

    color_jitter = getattr(data_cfg, 'color_jitter', True)
    brightness = getattr(data_cfg, 'brightness', 0.4)
    contrast = getattr(data_cfg, 'contrast', 0.4)
    saturation = getattr(data_cfg, 'saturation', 0.2)
    hue = getattr(data_cfg, 'hue', 0.1)
    color_jitter_p = getattr(data_cfg, 'color_jitter_p', 0.8)

    # Multi-crop 数量配置
    num_global_crops = getattr(data_cfg, 'num_global_crops', 2)
    num_local_crops = getattr(data_cfg, 'num_local_crops', 8)

    dataset = SimpleImageDataset(
        data_cfg.data_path,
        global_transform=g_transform,
        local_transform=l_transform,
        color_jitter=color_jitter,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        p=color_jitter_p,
        image_size=global_crop_size,
        num_global_crops=num_global_crops,
        num_local_crops=num_local_crops,
    )
    # dataloader = DataLoader(dataset, batch_size=data_cfg.batch_size,
    #                         shuffle=True, num_workers=data_cfg.num_workers)
    dataloader = DataLoader(dataset,
                            batch_size=data_cfg.batch_size,
                            shuffle=True,
                            num_workers=0,  # Windows上设为0避免多进程死锁
                            pin_memory=True
                            )
    return dataloader


def gen_paired_dataloader(data_cfg):
    """
    生成配对的STEM数据加载器，用于第二阶段训练

    期望data_cfg包含:
        - clean_dirs: 干净图像文件夹路径列表（或单个路径）
        - noisy_dirs: 加噪图像文件夹路径列表（或单个路径）
        - batch_size: 批次大小
        - image_size: 图像尺寸（默认为1024）
    """
    # 从配置中获取图像尺寸，默认为1024
    image_size = getattr(data_cfg, 'image_size', 1024)

    # 只需要基本的resize和归一化，不需要随机裁剪
    transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 兼容新旧配置格式
    clean_dirs = getattr(data_cfg, 'clean_dirs', getattr(data_cfg, 'clean_dir', None))
    noisy_dirs = getattr(data_cfg, 'noisy_dirs', getattr(data_cfg, 'noisy_dir', None))

    if clean_dirs is None or noisy_dirs is None:
        raise ValueError("配置中必须指定 clean_dirs/clean_dir 和 noisy_dirs/noisy_dir")

    dataset = PairedSTEMDataset(
        clean_dirs=clean_dirs,
        noisy_dirs=noisy_dirs,
        transform=transform,
        image_size=image_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=0,  # Windows上设为0避免多进程死锁
        pin_memory=True
    )

    return dataloader
