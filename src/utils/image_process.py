import torch
from torchvision.transforms import v2
from PIL import Image
import os

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _get_normalization():
    """Returns normalization transform with ImageNet statistics."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ])


def _get_color_jittering(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8):
    """Returns color jittering transform with grayscale augmentation."""
    return v2.Compose([
        v2.RandomApply([
            v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)],
            p=p,
        ),
        v2.RandomGrayscale(p=0.2),
    ])


def _get_gaussian_blur(kernel_size=9, sigma=(0.1, 2.0), p=1.0):
    """Returns Gaussian blur transform."""
    return v2.RandomApply([
        v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)],
        p=p
    )


def _get_geo_augmentation(crop_size, crop_scale, horizontal_flips=True):
    """Returns geometric augmentation transform (crop + flip)."""
    return v2.Compose([
        v2.RandomResizedCrop(
            size=crop_size,
            scale=crop_scale,
            ratio=(0.75, 1.33),
            interpolation=v2.InterpolationMode.BICUBIC,
        ),
        v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
    ])


# def global_transform(
#         global_crop_size=(224, 224),
#         global_crop_scale=(0.32, 1.0),
#         horizontal_flips=True
# ):
    """
    Creates two global crop transforms with different augmentation strategies.

    Returns:
        List of two transforms:
        - v1: geo + color + gaussian blur (p=1.0) + normalization
        - v2: geo + color + gaussian blur (p=0.1) + solarization + normalization
    """
    # geo_augmentation = _get_geo_augmentation(
    #     global_crop_size, global_crop_scale, horizontal_flips)
    # color_jittering = _get_color_jittering()
    # normalization = _get_normalization()

    # # Extra transform 1: strong gaussian blur
    # extra_transform_1 = v2.Compose([
    #     _get_gaussian_blur(p=1.0),
    # ])

    # # Extra transform 2: weak gaussian blur + solarization
    # extra_transform_2 = v2.Compose([
    #     _get_gaussian_blur(p=0.1),
    #     v2.RandomSolarize(threshold=128, p=0.2),
    # ])

    # global_transform_v1 = v2.Compose([
    #     geo_augmentation, color_jittering, extra_transform_1, normalization
    # ])
    # global_transform_v2 = v2.Compose([
    #     geo_augmentation, color_jittering, extra_transform_2, normalization
    # ])

    # return [global_transform_v1, global_transform_v2]
def global_transform(image_size: int = 512):
    """Global crop transform: resize to image_size and normalize."""
    return v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def local_transform(
        local_crop_size: int = 256,
        local_crop_scale=(0.1, 0.5),
        horizontal_flips: bool = True,
):
    """Local crop transform: random crop to local_crop_size with augmentation and normalize."""
    return v2.Compose([
        v2.RandomResizedCrop(
            size=local_crop_size,
            scale=local_crop_scale,
            ratio=(0.75, 1.33),
            interpolation=v2.InterpolationMode.BICUBIC,
        ),
        v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
        v2.ToTensor(),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def load_images(image_dir: str):
    """Loads images from a directory. All images are converted to RGB (3 channels)."""
    return [
        Image.open(os.path.join(image_dir, f)).convert("RGB")
        for f in os.listdir(image_dir)
    ]


def resize_image_transform(image_size: tuple[int, int]):
    """Resizes an image to a given size."""
    return v2.Compose([
        v2.Resize(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])