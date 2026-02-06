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


@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print("Configuration:\n", cfg)
    seed_everything(42)

    model_cfg = cfg.model
    data_cfg = cfg.data

    backbone = instantiate(model_cfg.backbone).to("cuda")
    dino_head = instantiate(model_cfg.dino_head).to("cuda")
    ibot_head = instantiate(model_cfg.ibot_head).to("cuda")

    dino_loss = instantiate(cfg.loss.dino_loss).to("cuda")
    koleo_loss = instantiate(cfg.loss.koleo_loss).to("cuda")
    ibot_loss = instantiate(cfg.loss.ibot_loss).to("cuda")

    student = StudentSSLModel(
        backbone=copy.deepcopy(backbone),
        dino_head=copy.deepcopy(dino_head),
        ibot_head=copy.deepcopy(ibot_head),
    ).to("cuda")

    teacher = TeacherSSLModel(
        backbone=copy.deepcopy(backbone),
        dino_head=copy.deepcopy(dino_head),
        ibot_head=copy.deepcopy(ibot_head),
    ).to("cuda")
    teacher.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.train.lr)
    epochs = cfg.train.epochs

    dataloader = gen_dataloader(data_cfg)

    for _ in range(epochs):
        for images in dataloader:

            images = [img.to("cuda") for img in images]
            num_batch = images[0].size(0)

            global_imgs = torch.vstack(images[:2])
            local_imgs = torch.vstack(images[2:])

            with torch.no_grad():
                global_cls_teacher, global_dino_teacher, global_ibot_teacher = teacher(
                    global_imgs
                )
                local_cls_teacher, local_dino_teacher, local_ibot_teacher = teacher(
                    local_imgs
                )
            global_cls_student, global_dino_student, global_ibot_student = student(
                global_imgs
            )
            local_cls_student, local_dino_student, local_ibot_student = student(
                local_imgs
            )

            optimizer.zero_grad()
            global_dino_loss_value = dino_loss(
                global_dino_student.view(2, -1, global_dino_student.size(-1)),
                global_dino_teacher.view(2, -1, global_dino_teacher.size(-1)),
            )
            local_dino_loss_value = dino_loss(
                local_dino_student.view(8, -1, local_dino_student.size(-1)),
                global_dino_teacher.view(2, -1, global_dino_teacher.size(-1)),
            )

            koleo_loss_value = koleo_loss(
                global_cls_student.view(-1, global_cls_student.size(-1))
            )

            mask = torch.rand(2 * num_batch, 196).to("cuda") < 0.4

            ibot_loss_value = ibot_loss(
                global_ibot_student.view(2 * 4, -1, global_ibot_student.size(-1)),
                global_ibot_teacher.view(2 * 4, -1, global_ibot_teacher.size(-1)),
                mask,
            )
            print("Global DINO loss:", global_dino_loss_value.item())
            print("Local DINO loss:", local_dino_loss_value.item())
            print("KoLeo loss:", koleo_loss_value.item())
            print("iBOT loss:", ibot_loss_value.item())

            loss = (
                global_dino_loss_value
                + local_dino_loss_value
                + koleo_loss_value
                + ibot_loss_value
            )
            loss.backward()
            optimizer.step()
            # break
        # break


if __name__ == "__main__":
    main()
