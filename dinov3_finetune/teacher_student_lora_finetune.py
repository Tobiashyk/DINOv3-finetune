import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import os
from pathlib import Path
import argparse
import logging
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# For LoRA
from peft import LoraConfig, get_peft_model

# Set up logging
logging.basicConfig(level=logging.INFO)

class MIMAugmentation:
    """MIM augmentation: RandomResizedCrop for zooming into local structures."""
    def __init__(self, scale=(0.1, 0.5)):
        self.transform = transforms.RandomResizedCrop(
            size=(1024, 1024),
            scale=scale,
            ratio=(0.75, 1.33),
            interpolation=transforms.InterpolationMode.BILINEAR
        )

    def __call__(self, img):
        return self.transform(img)

class STEMDataset(Dataset):
    """Dataset for STEM images with MIM augmentation."""
    def __init__(self, root_dirs, transform=None, augmentation=None):
        self.image_paths = []
        for root_dir in root_dirs:
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                self.image_paths.extend(list(Path(root_dir).glob(ext)))

        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Apply MIM augmentation
        if self.augmentation:
            augmented_img = self.augmentation(img)
        else:
            augmented_img = img

        if self.transform:
            # img = self.transform(img)
            view_tensor = self.transform(augmented_img)

        return view_tensor

def apply_mask(img_tensor, mask_ratio=0.5):
    """Apply random mask to image tensor."""
    batch_size, channels, height, width = img_tensor.shape

    # Create random mask at patch level
    patch_size = 16
    h_patches = height // patch_size
    w_patches = width // patch_size

    # Randomly mask patches
    mask = torch.rand(batch_size, h_patches, w_patches, device=img_tensor.device) > mask_ratio
    mask = mask.unsqueeze(1).float()  # [batch_size, 1, h_patches, w_patches]

    # Upsample mask to image size
    mask = torch.nn.functional.interpolate(mask, size=(height, width), mode='nearest')
    mask = mask.expand(-1, channels, -1, -1)  # Expand to all channels

    masked_img = img_tensor * mask
    return masked_img, mask

class DINOHead(nn.Module):
    """DINO head for global semantic learning from CLS token."""
    def __init__(self, emb_dim, out_dim=65536, use_bn=False, norm_last_layer=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(out_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class MIMHead(nn.Module):
    """MIM head for local feature learning from patch tokens."""
    def __init__(self, emb_dim):
        super().__init__()
        self.head = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        return self.head(x)

class StudentModel(nn.Module):
    """Student model with LoRA encoder and dual heads."""
    def __init__(self, encoder, dino_head, mim_head):
        super().__init__()
        self.encoder = encoder
        self.dino_head = dino_head
        self.mim_head = mim_head

    def forward(self, x):
        features = self.encoder.forward_features(x)
        cls_token = features['x_norm_clstoken']  # [batch_size, emb_dim]
        patch_tokens = features['x_norm_patchtokens']  # [batch_size, num_patches, emb_dim]

        # DINO head for CLS token
        dino_output = self.dino_head(cls_token)

        # MIM head for patch tokens
        mim_output = self.mim_head(patch_tokens)

        return dino_output, mim_output, patch_tokens

def get_dino_model(model_name='dinov3_vits16plus', weights_path=None):
    """Load DINOv3 model."""
    repo_dir = '../dinov3'
    model = torch.hub.load(repo_dir, model_name, source='local', weights=weights_path)
    return model

def apply_lora_to_dino(model, lora_config):
    """Apply LoRA to DINO model using peft."""
    config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=["qkv"],
        lora_dropout=lora_config['lora_dropout'],
        bias="none",
        modules_to_save=[],
    )

    peft_model = get_peft_model(model, config)

    # Freeze all parameters except LoRA
    for name, param in peft_model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    return peft_model

def update_teacher_ema(student, teacher, ema_decay=0.996):
    """Update teacher with EMA from student."""
    with torch.no_grad():
        for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
            teacher_param.data = ema_decay * teacher_param.data + (1 - ema_decay) * student_param.data

def dino_loss(student_cls, teacher_cls, temperature=0.1):
    """DINO loss: Cross-entropy between student and teacher CLS outputs."""
    # student_cls and teacher_cls: [batch_size, out_dim]

    # Compute logits
    teacher_logits = teacher_cls / temperature
    student_logits = student_cls / temperature

    # Cross-entropy loss: student learns to predict teacher
    loss = nn.functional.cross_entropy(student_logits, teacher_logits.argmax(dim=-1))

    return loss

def ibot_loss(student_patches, teacher_patches, mask_patches):
    """iBOT loss: MSE between student patches and teacher patches for masked regions."""
    # student_patches: [batch_size, num_patches, emb_dim]
    # teacher_patches: [batch_size, num_patches, emb_dim]
    # mask_patches: [batch_size, num_patches] (binary mask indicating which patches are masked)

    # Compute MSE loss only for masked patches
    diff = (student_patches - teacher_patches) ** 2  # [batch_size, num_patches, emb_dim]
    masked_diff = diff * mask_patches.unsqueeze(-1)  # Apply mask

    # Sum over embedding dimension and batch
    loss = masked_diff.sum() / (mask_patches.sum() + 1e-8)

    return loss

def train_epoch(student, teacher, dataloader, optimizer, device, ema_decay, temperature):
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_dino_loss = 0.0
    total_ibot_loss = 0.0
    step_count = 0

    # for teacher_imgs, student_imgs in dataloader:
    #     teacher_imgs = teacher_imgs.to(device)
    #     student_imgs = student_imgs.to(device)
    for images in dataloader:
        images = images.to(device)

        # Apply random mask to student input
        masked_imgs, mask = apply_mask(images, mask_ratio=0.5)

        optimizer.zero_grad()

        # Teacher forward pass (no gradients, using full images)
        with torch.no_grad():
            teacher_dino, teacher_mim, teacher_patches = teacher(images)

        # Student forward pass (using masked images)
        student_dino, student_mim, student_patches = student(masked_imgs)

        # Compute iBOT loss only (masked patch feature alignment)
        # Create patch-level mask from image mask
        patch_size = 16
        h_patches = mask.shape[2] // patch_size
        w_patches = mask.shape[3] // patch_size

        # Average across channels first, then pool spatially
        mask_avg_channels = mask.float().mean(dim=1, keepdim=True)  # [batch_size, 1, H, W]
        mask_patches = torch.nn.functional.avg_pool2d(mask_avg_channels, patch_size)  # [batch_size, 1, h_patches, w_patches]
        mask_patches = mask_patches.view(mask.shape[0], -1)  # [batch_size, num_patches]

        loss_ibot = ibot_loss(student_mim, teacher_mim, mask_patches)

        # Use only iBOT loss
        loss = loss_ibot

        loss.backward()
        optimizer.step()

        # Update teacher EMA
        update_teacher_ema(student, teacher, ema_decay)

        total_loss += loss.item()
        total_ibot_loss += loss_ibot.item()
        step_count += 1

    avg_loss = total_loss / step_count
    avg_ibot_loss = total_ibot_loss / step_count

    return avg_loss, avg_ibot_loss

def validate_and_visualize(student, test_dir, output_dir, device):
    """Validate by processing Sim_2H_test images and creating PCA visualizations."""
    student.eval()

    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_images = list(Path(test_dir).glob('*.png')) + list(Path(test_dir).glob('*.jpg'))

    for img_path in test_images:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        new_w = (w // 16) * 16
        new_h = (h // 16) * 16
        img = img.resize((new_w, new_h), Image.BILINEAR)

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = student.encoder.forward_features(img_tensor)
            patch_features = features['x_norm_patchtokens'].squeeze(0)

        h_grid = new_h // 16
        w_grid = new_w // 16
        patch_features_flat = patch_features.view(-1, patch_features.shape[-1]).cpu().numpy()

        pca = PCA(n_components=4)
        pca_result = pca.fit_transform(patch_features_flat)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()

        for i in range(4):
            component = pca_result[:, i]
            heatmap = component.reshape(h_grid, w_grid)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            axs[i].imshow(heatmap, cmap='inferno')
            axs[i].set_title(f"PCA Component {i+1}")
            axs[i].axis('off')

        output_path = os.path.join(output_dir, f"pca_{img_path.name}")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        logging.info(f"Processed {img_path.name}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    augmentation = MIMAugmentation(scale=(0.1, 0.5))

    # Dataset
    train_dirs = ['../PCA/train_pic/Sim_huge_1T', '../PCA/train_pic/Sim_huge_2H']
    dataset = STEMDataset(train_dirs, transform=transform, augmentation=augmentation)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Load Student encoder with LoRA
    student_encoder_base = get_dino_model(weights_path=args.weights_path)
    lora_config = {
        'r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout
    }
    student_encoder = apply_lora_to_dino(student_encoder_base, lora_config)

    # Create heads
    emb_dim = student_encoder_base.num_features
    dino_head = DINOHead(emb_dim=emb_dim, out_dim=args.dino_out_dim)
    mim_head = MIMHead(emb_dim=emb_dim)

    # Create student model
    student = StudentModel(student_encoder, dino_head, mim_head)
    student.to(device)

    # Initialize teacher as copy of student (EMA initialization)
    teacher_encoder = apply_lora_to_dino(get_dino_model(weights_path=args.weights_path), lora_config)
    teacher_dino_head = DINOHead(emb_dim=emb_dim, out_dim=args.dino_out_dim)
    teacher_mim_head = MIMHead(emb_dim=emb_dim)
    teacher = StudentModel(teacher_encoder, teacher_dino_head, teacher_mim_head)
    teacher.to(device)

    # Copy student weights to teacher initially
    teacher.load_state_dict(student.state_dict())
    teacher.eval()

    # Freeze teacher parameters (only updated via EMA)
    for param in teacher.parameters():
        param.requires_grad = False

    # Optimizer (only optimize student parameters)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    for epoch in range(args.epochs):
        loss, ibot_loss = train_epoch(student, teacher, dataloader, optimizer, device,
                                      args.ema_decay, args.temperature)
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}, iBOT: {ibot_loss:.4f}")

        if (epoch + 1) % args.save_freq == 0:
            # Save student weights
            student.encoder.save_pretrained(f"student_weights/student_encoder_epoch_{epoch+1}")
            torch.save(dino_head.state_dict(), f"student_weights/dino_head_epoch_{epoch+1}.pth")
            torch.save(mim_head.state_dict(), f"student_weights/mim_head_epoch_{epoch+1}.pth")

    # Final save
    student.encoder.save_pretrained("student_weights/student_encoder_final")
    torch.save(dino_head.state_dict(), "student_weights/dino_head_final.pth")
    torch.save(mim_head.state_dict(), "student_weights/mim_head_final.pth")

    # Validation
    logging.info("Starting validation with PCA visualization...")
    validate_and_visualize(student, '../PCA/train_pic/Sim_2H_test', 'pca_results', device)
    logging.info("Training and validation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher-Student DINO + iBOT LoRA Finetuning for DINOv3")
    parser.add_argument("--weights_path", type=str, default="../dinov3/weight/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
    parser.add_argument("--batch_size", type=int, default=2)  # Smaller batch size
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)  # Smaller LoRA rank
    parser.add_argument("--lora_alpha", type=int, default=16)  # Smaller LoRA alpha
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--ema_decay", type=float, default=0.996)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dino_out_dim", type=int, default=8192)  # Much smaller output dimension
    parser.add_argument("--save_freq", type=int, default=20)

    args = parser.parse_args()
    main(args)
