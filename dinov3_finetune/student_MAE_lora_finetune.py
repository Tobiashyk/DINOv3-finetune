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
    def __init__(self, scale=(0.05, 0.3)):  # More aggressive zooming
        self.transform = transforms.RandomResizedCrop(
            size=(1024, 1024),  # Original size
            scale=scale,
            ratio=(0.75, 1.33),  # Allow some aspect ratio variation
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
            img = self.transform(img)
            augmented_img = self.transform(augmented_img)

        return img, augmented_img

def apply_mask(img_tensor, mask_ratio=0.5):
    """Apply random mask to image tensor."""
    batch_size, channels, height, width = img_tensor.shape

    # Create mask
    mask = torch.rand(batch_size, 1, height, width, device=img_tensor.device) > mask_ratio
    mask = mask.float()

    # Apply mask
    masked_img = img_tensor * mask

    return masked_img, mask

class MIMDecoder(nn.Module):
    """Simple linear decoder for MIM reconstruction at patch level."""
    def __init__(self, emb_dim, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.decoder = nn.Linear(emb_dim, patch_size * patch_size * 3, bias=True)

    def forward(self, x):
        # x: [batch_size, num_patches, emb_dim]
        # Decode to patches: [batch_size, num_patches, patch_size*patch_size*3]
        patches = self.decoder(x)
        return patches

class MIMStudent(nn.Module):
    """Student model with LoRA encoder and linear decoder."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # Encoder forward
        features = self.encoder.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']  # [batch_size, num_patches, emb_dim]

        # Decoder forward
        reconstructed = self.decoder(patch_tokens)

        return reconstructed

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
    """Update teacher with EMA."""
    with torch.no_grad():
        for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
            teacher_param.data = ema_decay * teacher_param.data + (1 - ema_decay) * student_param.data

def mim_loss(reconstructed_patches, target_img, mask):
    """MSE loss for MIM reconstruction at patch level."""
    batch_size = target_img.shape[0]
    patch_size = 16
    img_size = 1024
    num_patches_per_dim = img_size // patch_size

    # Reshape target image to patches: [batch_size, 3, img_size, img_size] -> [batch_size, num_patches, 3*patch_size*patch_size]
    target_patches = target_img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    target_patches = target_patches.contiguous().view(batch_size, 3, num_patches_per_dim, num_patches_per_dim, patch_size, patch_size)
    target_patches = target_patches.permute(0, 2, 3, 4, 5, 1).contiguous()
    target_patches = target_patches.view(batch_size, -1, 3 * patch_size * patch_size)

    # Reshape mask to patches (expand to 3 channels to match RGB)
    mask_patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    mask_patches = mask_patches.contiguous().view(batch_size, 1, num_patches_per_dim, num_patches_per_dim, patch_size, patch_size)
    mask_patches = mask_patches.permute(0, 2, 3, 4, 5, 1).contiguous()
    mask_patches = mask_patches.view(batch_size, -1, patch_size * patch_size)
    # Expand mask to 3 channels
    mask_patches = mask_patches.unsqueeze(-1).expand(-1, -1, -1, 3).reshape(batch_size, -1, 3 * patch_size * patch_size)

    # Only compute loss on masked patches
    loss = nn.functional.mse_loss(reconstructed_patches * mask_patches, target_patches * mask_patches, reduction='mean')
    return loss

def train_epoch(student, teacher, dataloader, optimizer, device, ema_decay):
    student.train()

    total_loss = 0.0
    step_count = 0

    for teacher_imgs, student_imgs in dataloader:
        teacher_imgs = teacher_imgs.to(device)
        student_imgs = student_imgs.to(device)

        # Apply random mask to student input
        masked_student_imgs, mask = apply_mask(student_imgs, mask_ratio=0.5)

        optimizer.zero_grad()

        # Student forward pass (reconstruction)
        reconstructed = student(masked_student_imgs)

        # Compute MIM loss
        loss = mim_loss(reconstructed, student_imgs, mask)

        loss.backward()
        optimizer.step()

        # Update teacher EMA
        update_teacher_ema(student, teacher, ema_decay)

        total_loss += loss.item()
        step_count += 1

    return total_loss / step_count


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    augmentation = MIMAugmentation(scale=(0.05, 0.3))

    # Dataset
    train_dirs = ['../PCA/train_pic/Sim_1T', '../PCA/train_pic/Sim_2H','../PCA/train_pic/Sim_pos_1T','../PCA/train_pic/Sim_pos_2H']
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

    # Create decoder
    emb_dim = student_encoder_base.num_features
    decoder = MIMDecoder(emb_dim=emb_dim, patch_size=16)

    # Create student model
    student = MIMStudent(student_encoder, decoder)
    student.to(device)

    # Initialize teacher as copy of student
    teacher = MIMStudent(student_encoder_base, decoder)
    teacher.to(device)
    teacher.eval()

    # Optimizer (only optimize student parameters)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    for epoch in range(args.epochs):
        loss = train_epoch(student, teacher, dataloader, optimizer, device, args.ema_decay)
        logging.info(".4f")

        if (epoch + 1) % args.save_freq == 0:
            student.encoder.save_pretrained(f"student_weights/student_encoder_epoch_{epoch+1}")
            torch.save(decoder.state_dict(), f"student_weights/decoder_epoch_{epoch+1}.pth")

    # Final save
    student.encoder.save_pretrained("student_weights/student_encoder_final")
    torch.save(decoder.state_dict(), "student_weights/decoder_final.pth")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMA Teacher + MIM LoRA Finetuning for DINOv3")
    parser.add_argument("--weights_path", type=str, default="../dinov3/weight/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
    parser.add_argument("--batch_size", type=int, default=4)  # Smaller batch size for stability
    parser.add_argument("--epochs", type=int, default=10)  # More epochs
    parser.add_argument("--lr", type=float, default=5e-4)  # Higher learning rate
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)  # Higher LoRA rank
    parser.add_argument("--lora_alpha", type=int, default=32)  # Higher LoRA alpha
    parser.add_argument("--lora_dropout", type=float, default=0.05)  # Lower dropout
    parser.add_argument("--ema_decay", type=float, default=0.99)  # Faster teacher update
    parser.add_argument("--save_freq", type=int, default=20)  # Save less frequently

    args = parser.parse_args()
    main(args)
