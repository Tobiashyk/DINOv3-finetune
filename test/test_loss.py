import torch
import random
import numpy as np
from src.model.ssl import TeacherSSLModel, StudentSSLModel
from src.model.data import gen_dataloader

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
        lr=cfg.train.lr
    )
    epochs = cfg.train.epochs

    dataloader = gen_dataloader(data_cfg)
    
    for _ in range(epochs):
        for images in dataloader:
            images = [img.to('cuda') for img in images]
            global_imgs = torch.vstack(images[:2])
            local_imgs = torch.vstack(images[2:])

            with torch.no_grad():
                teacher_probs = teacher(global_imgs)
            student_probs = student(local_imgs)

            teacher_probs = teacher_probs.view(2, -1, teacher_probs.size(-1))
            student_probs = student_probs.view(8, -1, student_probs.size(-1))

            optimizer.zero_grad()
            dino_loss = loss(student_probs, teacher_probs)
            print("DINO loss:", dino_loss.item())
            dino_loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
