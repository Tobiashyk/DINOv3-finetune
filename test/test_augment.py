import os
from PIL import Image
from torchvision.utils import save_image
from src.utils.image_process import global_transform, local_transform


def main():
    os.makedirs("outputs/augment_test", exist_ok=True)

    test_image = Image.open("data/koala_test/koala_0.png")
    global_transforms = global_transform(
        global_crop_size=(224, 224),
        global_crop_scale=(0.32, 1.0),
        horizontal_flips=True
    )
    local_transforms = local_transform(
        local_crop_size=(96, 96),
        local_crop_scale=(0.05, 0.32),
        horizontal_flips=True,
        num_local_crops=8
    )
    transforms = global_transforms + local_transforms
    for i, transform in enumerate(transforms):
        tensor = transform(test_image)
        save_image(tensor, f"outputs/augment_test/transformed_image_{i}.png")


if __name__ == "__main__":
    main()
