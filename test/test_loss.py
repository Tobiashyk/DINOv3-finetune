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

    dino_head_dim = model_cfg.dino_head.out_dim
    ibot_head_dim = model_cfg.ibot_head.out_dim

    num_batch = data_cfg.batch_size

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

            global_imgs = torch.vstack(images[:2])
            local_imgs = torch.vstack(images[2:])


            ibot_mask = torch.rand(2 * num_batch, 196).to("cuda") < 0.4
            teacher_output = teacher(
                [global_imgs, local_imgs], masks=[None, None])
            student_output = student(
                [global_imgs, local_imgs], masks=[ibot_mask, None])

            print(student_output['global_cls_features'].shape)
            print(student_output['global_dino_output'].shape)
            print(student_output['global_ibot_output'].shape)
            print(student_output['local_cls_features'].shape)
            print(student_output['local_dino_output'].shape)
            print(student_output['local_ibot_output'].shape)
            print(student_output['masks'].shape)

0
            optimizer.zero_grad()
            global_dino_loss_value = dino_loss(
                student_output['global_dino_output'].view(2, -1, dino_head_dim),
                teacher_output['global_dino_output'].view(2, -1, dino_head_dim),
            )
            local_dino_loss_value = dino_loss(
                student_output['local_dino_output'].view(8, -1, dino_head_dim),
                teacher_output['local_dino_output'].view(8, -1, dino_head_dim),
            )

            koleo_loss_value = koleo_loss(
                student_output['global_cls_features'].view(-1, student_output['global_cls_features'].size(-1))
            )

            
            ibot_loss_value = ibot_loss(
                student_output['global_ibot_output'].view(2 * num_batch, -1, ibot_head_dim),
                teacher_output['global_ibot_output'].view(2 * num_batch, -1, ibot_head_dim),
                ibot_mask,
            )
            # print("Global DINO loss:", global_dino_loss_value.item())
            # print("Local DINO loss:", local_dino_loss_value.item())
            # print("KoLeo loss:", koleo_loss_value.item())
            # print("iBOT loss:", ibot_loss_value.item())

            loss = (
                global_dino_loss_value
                + local_dino_loss_value
                + koleo_loss_value
                + ibot_loss_value
            )
            print("Loss:", loss.item())
            loss.backward()
            optimizer.step()
            break
        break


if __name__ == "__main__":
    main()
