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
    ema_momentum = 0.996  # EMA momentum for teacher update

    dataloader = gen_dataloader(data_cfg)

    for _ in range(epochs):
        for images in dataloader:
            images = [img.to("cuda") for img in images]

            global_imgs = torch.vstack(images[:2])
            local_imgs = torch.vstack(images[2:])


            ibot_mask = torch.rand(2 * num_batch, 196).to("cuda") < 0.4
            teacher_output = teacher(
                [global_imgs, local_imgs], masks=[ibot_mask, None], teacher_temp=0.04)
            student_output = student(
                [global_imgs, local_imgs], masks=[ibot_mask, None])

            print("Student outputs:")
            print(f"  global_cls_features: {student_output['global_cls_features'].shape}")
            print(f"  global_dino_output: {student_output['global_dino_output'].shape}")
            print(f"  global_ibot_output: {student_output['global_ibot_output'].shape}")
            print(f"  local_cls_features: {student_output['local_cls_features'].shape}")
            print(f"  local_dino_output: {student_output['local_dino_output'].shape}")
            print(f"  local_ibot_output: {student_output['local_ibot_output'].shape}")
            print(f"  masks: {student_output['masks'].shape}")

            print("\nTeacher outputs (with Sinkhorn-Knopp centering):")
            print(f"  global_dino_centered: {teacher_output['global_dino_centered'].shape}")
            print(f"  local_dino_centered: {teacher_output['local_dino_centered'].shape}")
            print(f"  global_ibot_centered: {teacher_output['global_ibot_centered'].shape}")
            print(f"  local_ibot_centered: {teacher_output['local_ibot_centered'].shape}")

            optimizer.zero_grad()
            # Use centered teacher outputs for DINO loss
            global_dino_loss_value = dino_loss(
                student_output['global_dino_output'].view(2, -1, dino_head_dim),
                teacher_output['global_dino_centered'].view(2, -1, dino_head_dim),
            )
            local_dino_loss_value = dino_loss(
                student_output['local_dino_output'].view(8, -1, dino_head_dim),
                teacher_output['local_dino_centered'].view(8, -1, dino_head_dim),
            )

            # Use centered teacher outputs for iBOT loss
            ibot_loss_value = ibot_loss(
                student_output['global_ibot_output'].view(2 * num_batch, -1, ibot_head_dim),
                teacher_output['global_ibot_centered'].view(2 * num_batch, -1, ibot_head_dim),
                ibot_mask,
            )

            loss = (
                global_dino_loss_value
                + local_dino_loss_value
                + ibot_loss_value
            )
            print("Loss:", loss.item())
            loss.backward()
            optimizer.step()

            # Update teacher with EMA
            teacher.update_from_student_ema(student, momentum=ema_momentum)
            print(f"Teacher updated with EMA (momentum={ema_momentum})")
            break
        break


if __name__ == "__main__":
    main()
