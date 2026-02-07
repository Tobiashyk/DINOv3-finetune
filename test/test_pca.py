import torch
from src.utils.image_process import load_images, resize_image_transform
from src.utils.visualize import pca_transform_features

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")


@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    model_cfg = cfg.model
    data_cfg = cfg.data
    backbone = instantiate(model_cfg.backbone).to("cuda")
    print(backbone)

    images = load_images("data/koala_test")
    transform = resize_image_transform([data_cfg.image_w, data_cfg.image_h])

    images_transformed = [transform(img) for img in images]
    print(images_transformed[0])
    images_transformed = torch.stack(images_transformed).to("cuda")

    with torch.no_grad():
        outputs = backbone.forward_features(images_transformed)
    patch_features = outputs["x_norm_patchtokens"]

    patch_h = data_cfg.image_h // model_cfg.patch_size
    patch_w = data_cfg.image_w // model_cfg.patch_size
    pca_images = pca_transform_features(patch_features, patch_h, patch_w)
    for i, image in enumerate(pca_images):
        plt.imshow(image.squeeze(), aspect="auto")
        plt.axis("off")
        plt.savefig(
            f"outputs/pca/pca_visualization_{i}.png", bbox_inches="tight", pad_inches=0
        )
        plt.close()


if __name__ == "__main__":
    main()
