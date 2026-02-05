from PIL import Image
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'dinov3'))

from dinov3.data.augmentations import DataAugmentationDINO
from torchvision.utils import save_image


def main():
    test_image = Image.open("data/koala_test/koala_0.png")
    augmenter = DataAugmentationDINO(
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
        local_crops_number=8
    )
    outputs = augmenter(test_image)

    images = []
    images+=outputs['global_crops']  # first global crop
    images+=outputs["local_crops"]  # second global crop
    for i, img in enumerate(images):
        save_image(img, f"outputs/augment_test/output_augmented_{i}.png")


if __name__ == "__main__":
    main()
