import torch
import random
import numpy as np
from src.utils.image_process import make_transform
from src.data.test_datamodule import load_images
from src.model.ssl import TeacherSSLModel, StudentSSLModel

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import copy


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(
    config_path="../config", config_name="train", version_base=None
)
def main(cfg: DictConfig):
    print("Configuration:\n", cfg)
    seed_everything(42)

    model_cfg = cfg.model
    data_cfg = cfg.data

    backbone = instantiate(model_cfg.backbone).to('cuda')
    head = instantiate(model_cfg.head).to('cuda')
    loss = instantiate(cfg.loss).to('cuda')


    student = StudentSSLModel(
        backbone=copy.deepcopy(backbone),
        head=copy.deepcopy(head),
    ).to('cuda')

    teacher = TeacherSSLModel(
        backbone=copy.deepcopy(backbone),
        head=copy.deepcopy(head),
    ).to('cuda')
    teacher.requires_grad_(False)


    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=1e-3
    )

    images = load_images(data_cfg.path)
    transform = make_transform(data_cfg.image_w, data_cfg.image_h)

    images_transformed = [transform(img) for img in images]
    images_transformed = torch.stack(images_transformed).to('cuda')

    for _ in range(10):
        with torch.no_grad():
            teacher_probs = teacher(images_transformed)
        student_probs = student(images_transformed)

        optimizer.zero_grad()
        dino_loss = loss(student_probs, teacher_probs)
        print("DINO loss:", dino_loss.item())
        dino_loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
