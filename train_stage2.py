"""
第二阶段训练脚本：MAE风格的掩码去噪特征蒸馏（无iBOT head）

核心特点：
- Teacher模型完全冻结，只输入干净原图
- Student模型输入被遮挡的加噪图
- 直接使用backbone的patch tokens进行特征蒸馏（无iBOT head）
- Loss: MSE + Cosine Similarity on normalized patch tokens
"""

import torch
import torch.nn.functional as F
import random
import numpy as np
from pathlib import Path

from src.model.ssl import apply_mask
from src.model.data import gen_paired_dataloader

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from peft import PeftModel

import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mae_denoising_loss(student_patches, teacher_patches, mask=None, lambda_cosine=0.5):
    """
    MAE风格的掩码去噪损失
    直接在patch tokens上计算MSE + Cosine Similarity

    Args:
        student_patches: Student输出的patch tokens [B, N, D]
        teacher_patches: Teacher输出的patch tokens [B, N, D]
        mask: 可选的mask，用于只计算被mask区域的loss [B, N]
        lambda_cosine: cosine loss的权重
    Returns:
        loss: 标量损失值
    """
    B, N, D = student_patches.shape

    # L2归一化（对特征稳定性很重要）
    student_norm = F.normalize(student_patches, dim=-1)  # [B, N, D]
    teacher_norm = F.normalize(teacher_patches, dim=-1)  # [B, N, D]

    # 1. MSE Loss (在归一化后的特征上)
    mse_loss = F.mse_loss(student_norm, teacher_norm)

    # 2. Cosine Similarity Loss (1 - cos_sim)
    cos_sim = F.cosine_similarity(student_norm, teacher_norm, dim=-1)  # [B, N]

    if mask is not None:
        # 只计算被mask的patch的loss
        mask = mask.float()
        cos_loss = ((1 - cos_sim) * mask).sum() / (mask.sum() + 1e-8)
    else:
        cos_loss = (1 - cos_sim).mean()

    # 组合loss
    total_loss = mse_loss + lambda_cosine * cos_loss

    return total_loss, mse_loss, cos_loss


@hydra.main(config_path="./config", config_name="train_stage2", version_base=None)
def main(cfg: DictConfig):
    print("Stage 2 (MAE-style) Configuration:\n", cfg)
    seed_everything(cfg.train.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    model_cfg = cfg.model
    data_cfg = cfg.data

    # ========== 1. 加载第一阶段LoRA权重 ==========
    lora_weights_path = cfg.train.get('lora_weights_path', None)
    if lora_weights_path is None:
        raise ValueError("必须在配置中指定第一阶段的LoRA权重路径: train.lora_weights_path")

    logging.info(f"Loading Stage 1 LoRA weights from: {lora_weights_path}")

    # ========== 2. 创建Student和Teacher模型（只有Backbone，无iBOT head）==========
    logging.info("Creating Student and Teacher models (Backbone only, no iBOT head)...")

    # Student模型：加载LoRA权重
    student_backbone = instantiate(model_cfg.backbone).to(device)
    try:
        student_backbone = PeftModel.from_pretrained(student_backbone, lora_weights_path)
        logging.info("Student: Successfully loaded LoRA weights!")
    except Exception as e:
        logging.error(f"Error loading LoRA for Student: {e}")
        raise

    # Teacher模型：加载相同LoRA权重，完全冻结
    teacher_backbone = instantiate(model_cfg.backbone).to(device)
    try:
        teacher_backbone = PeftModel.from_pretrained(teacher_backbone, lora_weights_path)
        logging.info("Teacher: Successfully loaded LoRA weights!")
    except Exception as e:
        logging.error(f"Error loading LoRA for Teacher: {e}")
        raise

    teacher_backbone.eval()
    for param in teacher_backbone.parameters():
        param.requires_grad = False

    # 验证Teacher是否完全冻结
    teacher_trainable = sum(p.numel() for p in teacher_backbone.parameters() if p.requires_grad)
    logging.info(f"Teacher trainable parameters: {teacher_trainable} (should be 0)")

    # 验证Student的LoRA参数是否可训练
    logging.info("Checking Student LoRA parameters...")
    trainable_params = []
    for name, param in student_backbone.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            logging.info(f"  Trainable: {name}")
    if len(trainable_params) == 0:
        logging.warning("WARNING: No trainable parameters found! Trying to enable LoRA gradients...")
        # 手动启用LoRA参数
        for name, param in student_backbone.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                logging.info(f"  Enabled: {name}")

    # ========== 3. 优化器设置（只优化Student backbone的LoRA参数）==========
    lr = cfg.train.lr
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student_backbone.parameters()), lr, weight_decay=cfg.train.weight_decay)

    epochs = cfg.train.epochs
    mask_ratio = cfg.train.get('mask_ratio', 0.5)
    lambda_cosine = cfg.train.get('lambda_cosine', 0.5)

    # ========== 4. 数据加载器 ==========
    logging.info("Creating paired dataloader...")
    dataloader = gen_paired_dataloader(data_cfg)
    logging.info(f"Dataloader created. Dataset size: {len(dataloader.dataset)}, Batches: {len(dataloader)}")

    student_backbone.train()

    logging.info("=" * 60)
    logging.info("Starting Stage 2 Training (MAE-style Masked Denoising)")
    logging.info(f"Teacher: FROZEN (clean images)")
    logging.info(f"Student: TRAINABLE LoRA (masked noisy images)")
    logging.info(f"Architecture: Backbone only (NO iBOT head)")
    logging.info(f"Loss: MSE + {lambda_cosine} * CosineSimilarity on patch tokens")
    logging.info("=" * 60)

    # ========== 5. 训练循环 ==========
    for epoch in range(epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_cos = 0.0
        step_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for step, (clean_img, noisy_img) in enumerate(pbar):
            if step == 0:
                logging.info(f"First batch loaded. Clean shape: {clean_img.shape}, Noisy shape: {noisy_img.shape}")

            clean_img = clean_img.to(device)
            noisy_img = noisy_img.to(device)

            # ========== 第一步：混合域随机输入策略 ==========
            # 以50%概率选择noisy_img或clean_img作为Student输入
            if random.random() < 0.5:
                student_input = noisy_img
            else:
                student_input = clean_img

            # ========== 第二步：对选定的student_input施加掩码 ==========
            masked_student_input = apply_mask(student_input, mask_ratio=mask_ratio)

            # 清空梯度
            optimizer.zero_grad()

            # ========== Teacher前向传播（永远传入clean_img作为锚点）==========
            with torch.no_grad():
                teacher_features = teacher_backbone.forward_features(clean_img)
                teacher_patches = teacher_features['x_norm_patchtokens']  # [B, N, D]

            # ========== Student前向传播（传入masked_student_input）==========
            student_features = student_backbone.forward_features(masked_student_input)
            student_patches = student_features['x_norm_patchtokens']  # [B, N, D]

            # ========== 计算MAE风格的Denoising Loss ==========
            loss, mse_loss, cos_loss = mae_denoising_loss(
                student_patches,
                teacher_patches.detach(),
                mask=None,  # 可选：传入mask只计算被遮挡区域的loss
                lambda_cosine=lambda_cosine
            )

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_cos += cos_loss.item()
            step_count += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse_loss.item():.4f}',
                'cos': f'{cos_loss.item():.4f}'
            })

            # 每10个step输出日志
            if (step + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(dataloader)}] - "
                           f"Loss: {loss.item():.4f} (MSE: {mse_loss.item():.4f}, Cos: {cos_loss.item():.4f})")

        # 计算epoch平均损失
        avg_loss = total_loss / step_count
        avg_mse = total_mse / step_count
        avg_cos = total_cos / step_count

        logging.info(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f} "
                    f"(MSE: {avg_mse:.4f}, Cos: {avg_cos:.4f})")

        # 保存模型（只保存LoRA权重）
        save_interval = cfg.train.get('save_interval', 10)
        if (epoch + 1) % save_interval == 0:
            save_dir = Path('./student_weights') / cfg.train.save_path / f'stage2_mae_epoch_{epoch+1}'
            save_dir.mkdir(parents=True, exist_ok=True)

            # 只保存Student的LoRA权重（无iBOT head）
            student_backbone.save_pretrained(save_dir / 'encoder')

            logging.info(f"Model saved to {save_dir}")

    logging.info("Stage 2 (MAE-style) Training complete!")


if __name__ == "__main__":
    main()
