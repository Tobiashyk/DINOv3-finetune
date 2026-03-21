import logging
import argparse

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
import os


from pathlib import Path
from PIL import Image


logging.basicConfig(level=logging.INFO)

class PairedCropAugmentation:
    def __init__(self, 
                 scale=(0.2, 0.6),
                 n_crops=2):
        self.scale = scale
        self.n_crops = n_crops
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, clean_img, noisy_img):
        
        teacher_crops = []
        student_crops = []

        for _ in range(self.n_crops):
            i, j , h, w = transforms.RandomResizedCrop.get_params(
                clean_img, scale=self.scale, ratio=(0.75, 1.33)
            )

            clean_crop = TF.resized_crop(clean_img, i, j, h, w, size=(1024, 1024))
            teacher_crops.append(self.normalize(TF.to_tensor(clean_crop)))
            
            noisy_crop = TF.resized_crop(noisy_img, i, j, h, w, size=(1024, 1024))
            student_crops.append(self.normalize(TF.to_tensor(noisy_crop)))

        return teacher_crops, student_crops

class STEMDataset(Dataset):
    def __init__(self, clean_dirs, noisy_dirs, transform=None, augmentation=None):

        self.clean_dirs = [Path(d) for d in clean_dirs]
        self.noisy_dirs = [Path(d) for d in noisy_dirs]

        self.samples = []
        for c_dir, n_dir in zip(self.clean_dirs, self.noisy_dirs):
            c_files = sorted([f for f in os.listdir(c_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
            for fname in c_files:
                clean_path = c_dir / fname
                noisy_path = n_dir / fname
                if noisy_path.exists():
                    self.samples.append((clean_path, noisy_path))
                else:
                    print(f"Warning: Noisy counterpart not found for {clean_path}, skipping.")
                    
        print(f"Total paired samples loaded: {len(self.samples)}")
        
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        clean_path, noisy_path = self.samples[idx]
        clean_img = Image.open(clean_path).convert('RGB')
        noisy_img = Image.open(noisy_path).convert('RGB')
        clean_img = clean_img.resize((1024, 1024), Image.BILINEAR)
        noisy_img = noisy_img.resize((1024, 1024), Image.BILINEAR)

        if self.augmentation:
            teacher_crops, student_crops = self.augmentation(clean_img, noisy_img)

        if self.transform:
            teacher_full = self.transform(clean_img)
            student_full = self.transform(noisy_img)
        
        return teacher_crops, student_crops, teacher_full, student_full

class NoiseHead(nn.Module):
    def __init__(self, emb_dim, out_dim_noise, prototypes_noise, norm_last_layer=True):
        super().__init__()
        # mlp layers
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, out_dim_noise),
            nn.GELU(),
            nn.Linear(out_dim_noise, out_dim_noise)
        )

        # prototype layer
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim_noise, prototypes_noise, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class iBOTHead(nn.Module):
    # def __init__(self,emb_dim, out_dim_ibot, prototypes_ibot, norm_last_layer=True):
    #     super().__init__()

    #     # mlp layers
    #     self.mlp = nn.Sequential(
    #         nn.Linear(emb_dim, emb_dim),
    #         nn.GELU(),
    #         nn.Linear(emb_dim, out_dim_ibot)
    #     )

    #     # prototype layer
    #     self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim_ibot, prototypes_ibot, bias=False))
    #     self.last_layer.weight_g.data.fill_(1.0)
    #     if norm_last_layer:
    #         self.last_layer.weight_g.requires_grad = False
    
    # def forward(self, x):
    #     x = self.mlp(x)
    #     x = nn.functional.normalize(x, dim=-1, p=2)
    #     x = self.last_layer(x)
    #     return x

    def __init__(self, emb_dim):
        super().__init__()
        self.head = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        return self.head(x)

class StudentModel(nn.Module):
    def __init__(self, encoder, noise_head, ibot_head):
        super().__init__()
        self.encoder = encoder
        self.noise_head = noise_head
        self.ibot_head = ibot_head

    def forward(self, x):
        features = self.encoder.forward_features(x)
        cls_token = features['x_norm_clstoken']
        patch_tokens = features['x_norm_patchtokens']

        # Noise head for cls token
        noise_output = self.noise_head(cls_token)

        # iBOT head for patch tokens
        ibot_output = self.ibot_head(patch_tokens)

        return noise_output, ibot_output

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dino_model(model_name='dinov3_vits16plus', weights_path=None):
    repo_dir = '../dinov3'
    model = torch.hub.load(repo_dir, model_name, source='local', weights=weights_path)
    return model

def apply_lora_to_dino(model, lora_config):
    config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=["qkv"],
        lora_dropout=lora_config['lora_dropout'],
        bias="none",
        modules_to_save=[]
    )
    peft_model = get_peft_model(model, config)

    for name, param in peft_model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    return peft_model

def apply_mask(img, mask_ratio=0.5):
    batch_size, channels, height, width = img.shape

    # create mask
    patch_size = 16
    h_patches = height // patch_size
    w_patches = width // patch_size
    mask = torch.rand(batch_size, h_patches, w_patches, device=img.device) > mask_ratio
    # mask_indices = [torch.nonzero(~mask[b], as_tuple=False) for b in range(batch_size)]
    mask = mask.unsqueeze(1).float()  # [batch_size, 1, h_patches, w_patches]

    # Upsample mask to image size
    mask = torch.nn.functional.interpolate(mask, size=(height, width), mode='nearest')
    mask = mask.expand(-1, channels, -1, -1)  # Expand to all channels
    masked_img = img * mask

    # return masked_img, mask_indices
    return masked_img

def noise_loss(student_cls, teacher_cls, center, temperature=0.1):
    # compute logits
    student_logits = student_cls / temperature
    teacher_logits = (teacher_cls - center) / (temperature - 0.03)

    # compute probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)

    #compute loss
    loss = torch.sum(-teacher_probs * student_log_probs, dim=-1).mean()

    return loss

def ibot_loss(student_patches, teacher_patches, temperature=0.1):
    batch_size = student_patches.size(0)
    # h_patches = w_patches = int(student_patches.size(1) ** 0.5)
    total_loss = 0.0
    total_batches = 0

    for b in range(batch_size):
        
        # get masked patch tokens
        student_all = F.normalize(student_patches[b], dim=-1)  # [num_patches, emb_dim]
        teacher_all = F.normalize(teacher_patches[b], dim=-1)  # [num_patches, emb_dim]
        
        sim_matrix = torch.matmul(student_all, teacher_all.T) / temperature
        labels = torch.arange(student_patches.size(1), device=sim_matrix.device)

        loss = F.cross_entropy(sim_matrix, labels)

        total_loss += loss
        total_batches += 1
        
    ibot_loss = total_loss / total_batches
    return ibot_loss

def update_teacher_model(student_model, teacher_model, ema_decay):
    with torch.no_grad():
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
            teacher_param.data = ema_decay * teacher_param.data + (1 - ema_decay) * student_param.data

def train_epoch(student_model, teacher_model, dataloader, optimizer, device, ema_decay, temperature, center):
    student_model.train()
    teacher_model.eval()

    center_rate = 0.999
    total_loss = 0.0
    total_noise_loss = 0.0
    total_ibot_loss = 0.0
    step_count = 0

    for teacher_crops, student_crops, teacher_full, student_full in dataloader:

        teacher_crops = torch.cat([crop.to(device) for crop in teacher_crops])
        student_crops = torch.cat([crop.to(device) for crop in student_crops])
        teacher_full = teacher_full.to(device)
        student_full = student_full.to(device)

        # apply masking to local view
        # masked_global_view, mask_indices = apply_mask(global_view, mask_ratio=args.mask_ratio)
        masked_global_view = apply_mask(student_full, mask_ratio=args.mask_ratio)

        # reset gradients
        optimizer.zero_grad()

        # teacher forward pass
        with torch.no_grad():
            teacher_noise_output, _ = teacher_model(teacher_crops)
            _ , teacher_ibot_output = teacher_model(teacher_full)
        # student forward pass
        student_noise_output, _ = student_model(student_crops)
        _, student_ibot_output = student_model(masked_global_view)

        loss_noise = noise_loss(student_noise_output, teacher_noise_output, center, temperature)
        loss_ibot = ibot_loss(student_ibot_output, teacher_ibot_output, temperature)

        # total loss
        loss = args.lambda_noise * loss_noise + args.lambda_ibot * loss_ibot
        # loss = loss_noise + loss_ibot

        # backward pass and optimization
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        update_teacher_model(student_model, teacher_model, ema_decay)

        with torch.no_grad():
            # 计算当前 Batch Teacher 输出的均值
            batch_center = torch.mean(teacher_noise_output, dim=0, keepdim=True)
            # 滑动平均更新
            center = center * center_rate + batch_center * (1 - center_rate)

        total_loss += loss.item()
        total_noise_loss += loss_noise.item()
        total_ibot_loss += loss_ibot.item()
        step_count += 1

    avg_loss = total_loss / step_count
    avg_noise_loss = total_noise_loss / step_count
    avg_ibot_loss = total_ibot_loss / step_count

    return avg_loss, avg_noise_loss, avg_ibot_loss, center

def main(generator):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # data preprocessing
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    augmentation = PairedCropAugmentation(scale=(0.2, 0.6), n_crops=2)

    # Dataset
    clean_dirs = ['../PCA/train_pic/train_try/graphene_stem_dataset_AA/images_clean',
                   '../PCA/train_pic/train_try/graphene_stem_dataset_AB/images_clean',
                   '../PCA/train_pic/train_try/MoS2_stem_dataset_1T/images_clean',
                   '../PCA/train_pic/train_try/MoS2_stem_dataset_2H/images_clean']
    noisy_dirs = ['../PCA/train_pic/train_try/graphene_stem_dataset_AA/images_noisy_all',
                   '../PCA/train_pic/train_try/graphene_stem_dataset_AB/images_noisy_all',
                   '../PCA/train_pic/train_try/MoS2_stem_dataset_1T/images_noisy_all',
                   '../PCA/train_pic/train_try/MoS2_stem_dataset_2H/images_noisy_all']
    dataset = STEMDataset(clean_dirs, noisy_dirs, transform=transform, augmentation=augmentation)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4,
                            worker_init_fn=seed_worker,
                            generator=generator
                            )

    # Create student model
    student_encoder_base = get_dino_model(weights_path=args.weights_path)
    lora_config = {
        'r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout
    }

    student_encoder = apply_lora_to_dino(student_encoder_base, lora_config)
    emb_dim = student_encoder_base.num_features
    student_noise_head = NoiseHead(emb_dim=emb_dim, out_dim_noise=args.out_dim_noise, prototypes_noise=args.prototypes_noise)
    student_ibot_head = iBOTHead(emb_dim=emb_dim)
    student_model = StudentModel(student_encoder, student_noise_head, student_ibot_head)
    student_model.to(device)

    # Create teacher model
    teacher_encoder_base = get_dino_model(weights_path=args.weights_path)
    teacher_encoder = apply_lora_to_dino(teacher_encoder_base, lora_config)
    teacher_noise_head = NoiseHead(emb_dim=emb_dim, out_dim_noise=args.out_dim_noise, prototypes_noise=args.prototypes_noise)
    teacher_ibot_head = iBOTHead(emb_dim=emb_dim)
    teacher_model = StudentModel(teacher_encoder, teacher_noise_head, teacher_ibot_head)
    teacher_model.to(device)

    # # Create student and teacher models
    # student_encoder_base = get_dino_model(weights_path=args.weights_path)
    # emb_dim = student_encoder_base.num_features
    # student_encoder = apply_lora_to_dino(student_encoder_base, lora_config)
    # teacher_encoder_base = get_dino_model(weights_path=args.weights_path)
    # teacher_encoder = apply_lora_to_dino(teacher_encoder_base, lora_config)

    # student_ibot_head = iBOTHead(emb_dim=emb_dim)
    # student_dino_head = DINOHead(emb_dim=emb_dim, out_dim_dino=args.out_dim_dino, prototypes_dino=args.prototypes_dino)
    # teacher_dino_head = DINOHead(emb_dim=emb_dim, out_dim_dino=args.out_dim_dino, prototypes_dino=args.prototypes_dino)
    # teacher_ibot_head = iBOTHead(emb_dim=emb_dim)

    # student_model = StudentModel(student_encoder, student_dino_head, student_ibot_head)
    # student_model.to(device)

    # teacher_model = StudentModel(teacher_encoder, teacher_dino_head, teacher_ibot_head)
    # teacher_model.to(device)
    
    #  Initialize teacher weights = student weights
    teacher_model.load_state_dict(student_model.state_dict())
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Optimizer
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    center = torch.zeros(1, args.prototypes_noise).to(device)

    # Training loop
    for epoch in range(args.epochs):
        loss, loss_noise, loss_ibot, center = train_epoch(
            student_model, teacher_model, dataloader, optimizer, device, args.ema_decay, args.temperature, center=center
        )
        logging.info(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}, Noise Loss: {loss_noise:.4f}, iBOT Loss: {loss_ibot:.4f}')

        # save student weights
        if (epoch + 1) % args.save_interval == 0:
            student_model.encoder.save_pretrained('./student_weights/' + args.save_path + f'/student_encoder_epoch_{epoch+1}')
            torch.save(student_noise_head.state_dict(), './student_weights/' + args.save_path + f'/noise_head_epoch_{epoch+1}.pth')
            torch.save(student_ibot_head.state_dict(), './student_weights/' + args.save_path + f'/ibot_head_epoch_{epoch+1}.pth')

    logging.info("Training complete!")

if __name__ == "__main__":
    # for seed in [43, 123, 456, 789, 2026, 2077, 54321]:

    parser = argparse.ArgumentParser(description="Teacher-Student DINO + iBOT LoRA Finetune for dinov3")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weights_path", type=str, default='../dinov3/weight/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
    parser.add_argument("--save_path", type=str, default="noise_finetune_lora")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--out_dim_noise", type=int, default=384)
    parser.add_argument("--out_dim_ibot", type=int, default=384)
    parser.add_argument("--prototypes_noise", type=int, default=2048)
    parser.add_argument("--prototypes_ibot", type=int, default=384)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lambda_noise", type=float, default=2.0)
    parser.add_argument("--lambda_ibot", type=float, default=0.5)
    parser.add_argument("--ema_decay", type=float, default=0.996)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=10)

    args = parser.parse_args()

    generator = set_seed(42)
    main(generator)






