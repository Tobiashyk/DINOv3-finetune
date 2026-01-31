import torch
from torchvision.transforms import v2

def make_transform(resize_w = 256, resize_h = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_w, resize_h), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])