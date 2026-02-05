from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
import os

from src.utils.image_process import global_transform, local_transform


class SimpleImageDataset(Dataset):
    def __init__(self, image_dir: str, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        original_image = Image.open(img_path).convert('RGB')

        images = []
        for transform in self.transforms:
            transformed_image = transform(original_image)
            images.append(transformed_image)
        return images


def gen_dataloader(data_cfg):
    global_transforms = global_transform(
        global_crop_size=data_cfg.global_crop_size,
        global_crop_scale=data_cfg.global_crop_scale,
        horizontal_flips=data_cfg.horizontal_flips
    )
    local_transforms = local_transform(
        local_crop_size=data_cfg.local_crop_size,
        local_crop_scale=data_cfg.local_crop_scale,
        horizontal_flips=data_cfg.horizontal_flips,
        num_local_crops=data_cfg.num_local_crops
    )
    transforms = global_transforms + local_transforms

    dataset = SimpleImageDataset(data_cfg.data_path, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=data_cfg.batch_size,
                            shuffle=True, num_workers=data_cfg.num_workers)
    return dataloader
