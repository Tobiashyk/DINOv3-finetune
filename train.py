import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import random
import numpy as np
import csv
from src.model.ssl import StudentSSLModel, apply_lora_to_dino, generate_token_masks
from src.model.data import gen_dataloader
from src.loss.dino_clstoken_loss import dino_loss_calculate
from src.loss.ibot_patch_loss import iBOTPatchLoss
from src.loss.koleo_loss import KoLeoLoss

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # cuBLAS deterministic for torch.matmul
    torch.use_deterministic_algorithms(True)


@hydra.main(config_path="./config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print("Configuration:\n", cfg)
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    model_cfg = cfg.model
    data_cfg = cfg.data

    num_batch = data_cfg.batch_size

    # Multi-crop config
    num_global_crops = getattr(data_cfg, 'num_global_crops', 2)
    num_local_crops = getattr(data_cfg, 'num_local_crops', 8)
    logging.info(f"Multi-crop config: {num_global_crops} global crops + {num_local_crops} local crops")

    logging.info("Loading backbone model...")
    lora_config = {
        'r': cfg.train.lora_r,
        'lora_alpha': cfg.train.lora_alpha,
        'lora_dropout': cfg.train.lora_dropout
    }

    # Create student
    logging.info("Creating student model...")
    student_backbone = instantiate(model_cfg.backbone).to(device)
    student_backbone = apply_lora_to_dino(student_backbone, lora_config)
    student_dino_head = instantiate(model_cfg.dino_head, in_dim=student_backbone.num_features).to(device)
    student_ibot_head = instantiate(model_cfg.ibot_head, in_dim=student_backbone.num_features).to(device)

    # Create teacher (initialized as copy of student)
    teacher_backbone = instantiate(model_cfg.backbone).to(device)
    teacher_backbone = apply_lora_to_dino(teacher_backbone, lora_config)
    teacher_dino_head = instantiate(model_cfg.dino_head, in_dim=teacher_backbone.num_features).to(device)
    teacher_ibot_head = instantiate(model_cfg.ibot_head, in_dim=teacher_backbone.num_features).to(device)

    student = StudentSSLModel(
        backbone=student_backbone,
        dino_head=student_dino_head,
        ibot_head=student_ibot_head,
    ).to(device)

    teacher = StudentSSLModel(
        backbone=teacher_backbone,
        dino_head=teacher_dino_head,
        ibot_head=teacher_ibot_head,
    ).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    lr = cfg.train.lr * num_batch
    optimizer = torch.optim.AdamW(student.parameters(), lr, weight_decay=cfg.train.weight_decay)
    epochs = cfg.train.epochs
    ema_decay = cfg.train.ema_decay
    ema_decay = 1.0 - (1.0 - ema_decay) * num_batch
    student_temp = cfg.train.temperature
    teacher_temp = cfg.train.get('teacher_temp', 0.04)
    mask_ratio = cfg.train.mask_ratio
    center_momentum = 0.9

    # ──────────────────────────────────────────────────
    # DINO center (for CLS token loss)
    # ──────────────────────────────────────────────────
    dino_head_out_dim = model_cfg.dino_head.out_dim
    dino_center = torch.zeros(1, dino_head_out_dim, device=device)
    logging.info(f"DINO center: shape={dino_center.shape}, momentum={center_momentum}")

    # ──────────────────────────────────────────────────
    # iBOT patch loss module (SK + cross-entropy)
    # ──────────────────────────────────────────────────
    ibot_out_dim = model_cfg.ibot_head.out_dim
    ibot_loss_module = iBOTPatchLoss(
        patch_out_dim=ibot_out_dim,
        student_temp=student_temp,
        center_momentum=center_momentum,
    ).to(device)
    logging.info(f"iBOT loss module: patch_out_dim={ibot_out_dim}, student_temp={student_temp}")

    # ──────────────────────────────────────────────────
    # KoLeo loss (entropic regularization on CLS tokens)
    # ──────────────────────────────────────────────────
    koleo_loss_module = KoLeoLoss()
    koleo_weight = cfg.train.get('koleo_weight', 0.1)
    logging.info(f"KoLeo loss weight: {koleo_weight}")

    # Patch grid for token-level masking
    global_crop_size = getattr(data_cfg, 'global_crop_size', 512)
    local_crop_size = getattr(data_cfg, 'local_crop_size', 256)
    patch_size = getattr(model_cfg, 'patch_size', 16)
    n_patches = (global_crop_size // patch_size) ** 2
    logging.info(f"Crop sizes: global={global_crop_size}, local={local_crop_size}")
    logging.info(f"Token mask: n_patches={n_patches}, mask_ratio={mask_ratio}")

    logging.info("Creating dataloader...")
    dataloader = gen_dataloader(data_cfg)
    logging.info(f"Dataset size: {len(dataloader.dataset)}, Batches: {len(dataloader)}")

    student.train()
    teacher.eval()
    logging.info("Starting training...")

    save_dir = './student_weights/' + cfg.train.save_path
    os.makedirs(save_dir, exist_ok=True)
    csv_file = os.path.join(save_dir, 'training_losses.csv')

    for epoch in range(epochs):
        total_loss = 0.0
        total_dino_loss = 0.0
        total_ibot_loss = 0.0
        total_koleo_loss = 0.0
        step_count = 0

        if epoch == 0:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['global_step', 'epoch', 'step', 'dino_loss', 'ibot_loss', 'koleo_loss', 'total_loss'])
            logging.info(f"Created {csv_file} with headers")

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for step, (global_crops, local_crops) in enumerate(pbar):
            if step == 0:
                logging.info(
                    f"First batch — Global crops: {len(global_crops)} x {global_crops[0].shape}, "
                    f"Local crops: {len(local_crops)} x {local_crops[0].shape}"
                )

            global_crops = [g.to(device) for g in global_crops]
            local_crops = [lc.to(device) for lc in local_crops]

            B = global_crops[0].shape[0]

            # ── Token-level mask (shared across all global crops) ──────────────
            # mask[b, p] = True  →  patch p of sample b is replaced by mask_token
            # inside the backbone's prepare_tokens_with_masks()
            mask = generate_token_masks(B, n_patches, mask_ratio, device)  # [B, N]
            n_masked = mask.sum()  # scalar tensor

            optimizer.zero_grad()

            # ══════════════════════════════════════════════════════════════════
            # Teacher forward  (no masks — teacher always sees the clean image)
            # ══════════════════════════════════════════════════════════════════
            with torch.no_grad():
                teacher_dino_outputs = []
                teacher_ibot_outputs = []  # list of [B, N, ibot_dim]
                for g_crop in global_crops:
                    dino_out, ibot_out, _ = teacher(g_crop)   # masks=None
                    teacher_dino_outputs.append(dino_out)
                    teacher_ibot_outputs.append(ibot_out)
                # [num_global, B, dino_dim]
                teacher_dino_output = torch.stack(teacher_dino_outputs, dim=0)

            # ── Update iBOT center from ALL teacher patch logits ───────────────
            with torch.no_grad():
                # Stack all global crop teacher ibot outputs: [num_global, B, N, ibot_dim]
                all_teacher_ibot = torch.stack(teacher_ibot_outputs, dim=0)
                # Merge crops and batch for a unified center update: [num_global*B, N, ibot_dim]
                ibot_loss_module.update_center(all_teacher_ibot.flatten(0, 1))

            # ══════════════════════════════════════════════════════════════════
            # Student forward — global crops WITH mask tokens (DINO + iBOT)
            # ══════════════════════════════════════════════════════════════════
            student_dino_global_outputs = []
            student_ibot_outputs = []  # list of [B, N, ibot_dim]
            student_cls_tokens = []    # raw CLS tokens for KoLeo: list of [B, D]
            for g_crop in global_crops:
                dino_out, ibot_out, cls_tok = student(g_crop, masks=mask)
                student_dino_global_outputs.append(dino_out)
                student_ibot_outputs.append(ibot_out)
                student_cls_tokens.append(cls_tok)

            # Student forward — local crops WITHOUT masks (DINO only)
            student_dino_local_outputs = []
            for lc in local_crops:
                dino_out, _, cls_tok = student(lc)
                student_dino_local_outputs.append(dino_out)
                student_cls_tokens.append(cls_tok)

            # [num_global + num_local, B, dino_dim]
            student_dino_output = torch.stack(
                student_dino_global_outputs + student_dino_local_outputs, dim=0
            )

            # ══════════════════════════════════════════════════════════════════
            # DINO Loss  (CLS token cross-entropy, unchanged)
            # ══════════════════════════════════════════════════════════════════
            loss_dino = dino_loss_calculate(
                student_output=student_dino_output,
                teacher_output=teacher_dino_output,
                center=dino_center,
                student_temp=student_temp,
                teacher_temp=teacher_temp,
                ignore_diagonal=True,
            )

            # ══════════════════════════════════════════════════════════════════
            # iBOT Loss  (token-level masking + SK soft labels + cross-entropy)
            # ══════════════════════════════════════════════════════════════════
            # For each global crop:
            #   1. Flatten patch dim: [B, N, K] -> [B*N, K]
            #   2. Select only masked positions: [total_masked, K]
            #   3. SK-normalize teacher masked logits → soft labels
            #   4. Cross-entropy between student masked logits and teacher soft labels
            loss_ibot = torch.tensor(0.0, device=device)
            mask_flat = mask.flatten()  # [B*N] bool
            for i in range(num_global_crops):
                # Student masked logits
                student_masked = student_ibot_outputs[i].flatten(0, 1)[mask_flat]  # [total_masked, K]
                # Teacher masked logits (detached — teacher has no_grad context above)
                teacher_masked = teacher_ibot_outputs[i].flatten(0, 1)[mask_flat]  # [total_masked, K]

                # Sinkhorn-Knopp soft labels from teacher
                teacher_soft_labels = ibot_loss_module.sinkhorn_knopp_teacher(
                    teacher_masked.detach(), teacher_temp, n_masked
                )  # [total_masked, K]

                loss_ibot += ibot_loss_module.forward_masked(
                    student_masked_logits=student_masked,
                    teacher_soft_labels=teacher_soft_labels,
                    student_masks_flat=mask,
                )

            loss_ibot = loss_ibot / num_global_crops

            # ══════════════════════════════════════════════════════════════════
            # KoLeo Loss  (entropic regularization on backbone CLS tokens)
            # Concatenate all crops' CLS tokens: [(num_global+num_local)*B, D]
            # ══════════════════════════════════════════════════════════════════
            all_cls = torch.cat(student_cls_tokens, dim=0)  # [(n_global+n_local)*B, D]
            loss_koleo = koleo_loss_module(all_cls)

            # ══════════════════════════════════════════════════════════════════
            # Backward + optimise
            # ══════════════════════════════════════════════════════════════════
            loss = loss_dino + loss_ibot + koleo_weight * loss_koleo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            optimizer.step()

            # ── EMA teacher update ─────────────────────────────────────────────
            with torch.no_grad():
                for s_p, t_p in zip(student.parameters(), teacher.parameters()):
                    t_p.data = ema_decay * t_p.data + (1 - ema_decay) * s_p.data

            # ── DINO center update ─────────────────────────────────────────────
            with torch.no_grad():
                batch_center = teacher_dino_output.mean(dim=(0, 1), keepdim=True)  # [1, 1, dino_dim]
                batch_center = batch_center.squeeze(0)                              # [1, dino_dim]
                dino_center = dino_center * center_momentum + batch_center * (1 - center_momentum)

            total_loss += loss.item()
            total_dino_loss += loss_dino.item()
            total_ibot_loss += loss_ibot.item()
            total_koleo_loss += loss_koleo.item()
            step_count += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dino': f'{loss_dino.item():.4f}',
                'ibot': f'{loss_ibot.item():.4f}',
                'koleo': f'{loss_koleo.item():.4f}',
            })

            if (step + 1) % 20 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(dataloader)}] — "
                    f"Loss: {loss.item():.4f}, DINO: {loss_dino.item():.4f}, "
                    f"iBOT: {loss_ibot.item():.4f}, KoLeo: {loss_koleo.item():.4f}"
                )

            global_step = epoch * len(dataloader) + step + 1
            if global_step % 20 == 0:
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        global_step, epoch + 1, step + 1,
                        f'{loss_dino.item():.4f}', f'{loss_ibot.item():.4f}',
                        f'{loss_koleo.item():.4f}', f'{loss.item():.4f}'
                    ])
                student.encoder.save_pretrained(
                    os.path.join(save_dir, f'student_encoder_step_{global_step}')
                )

        avg_loss = total_loss / step_count
        avg_dino_loss = total_dino_loss / step_count
        avg_ibot_loss = total_ibot_loss / step_count
        avg_koleo_loss = total_koleo_loss / step_count
        logging.info(
            f"Epoch [{epoch+1}/{epochs}] — Avg Total: {avg_loss:.4f}, "
            f"Avg DINO: {avg_dino_loss:.4f}, Avg iBOT: {avg_ibot_loss:.4f}, "
            f"Avg KoLeo: {avg_koleo_loss:.4f}"
        )



    logging.info("Training complete!")


if __name__ == "__main__":
    main()
