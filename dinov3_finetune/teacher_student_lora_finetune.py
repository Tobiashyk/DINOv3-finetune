import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
                paths = sorted(list(Path(root_dir).glob(ext)))
                self.image_paths.extend(paths)

        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        

        # Global view (full image)
        global_view = img

        # Apply MIM augmentation for local view
        if self.augmentation:
            local_view = self.augmentation(img)
        else:
            local_view = img

        if self.transform:
            global_view = self.transform(global_view)
            local_view = self.transform(local_view)

        return global_view, local_view

def apply_mask(img_tensor, mask_ratio=0.5):
    """Apply random mask to image tensor at patch level."""
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

    # ===== Original implementation (commented out) =====
    # Compute logits
    # teacher_logits = teacher_cls / (temperature - 0.03)
    # student_logits = student_cls / temperature
    student_temp = temperature
    teacher_temp = temperature - 0.03

    # Cross-entropy loss: student learns to predict teacher
    # loss = nn.functional.cross_entropy(student_logits, teacher_logits.argmax(dim=-1))
    student_log_probs = F.log_softmax(student_cls / student_temp, dim=-1)
    teacher_probs = F.softmax((teacher_cls / teacher_temp).detach(), dim=-1)
    loss = torch.sum(-teacher_probs * student_log_probs, dim=-1).mean()

    # ===== New implementation with teacher centering and sharpening =====
    # student_temp = temperature
    # teacher_temp = temperature - 0.04  # Slightly lower temperature for sharpening

    # Apply centering: subtract the running average center from teacher output
    # teacher_centered = teacher_cls - teacher_center

    # Compute probabilities with sharpening (lower temperature)
    # student_log_probs = F.log_softmax(student_cls / student_temp, dim=-1)
    # teacher_probs = F.softmax(teacher_centered / teacher_temp, dim=-1)

    # Cross-entropy loss
    # loss = torch.sum(-teacher_probs * student_log_probs, dim=-1).mean()

    return loss

def ibot_loss(student_patches, teacher_patches, mask_patches, temperature=0.1):
    """iBOT loss: Cross-entropy between student and teacher patch tokens."""
    # student_patches: [batch_size, num_patches, emb_dim]
    # teacher_patches: [batch_size, num_patches, emb_dim]
    # mask_patches: [batch_size, num_patches] - boolean mask indicating which patches are masked

    # ===== Original implementation (commented out) =====
    # batch_size, num_patches, emb_dim = student_patches.shape

    # # Normalize features
    # student_norm = F.normalize(student_patches, dim=-1)
    # teacher_norm = F.normalize(teacher_patches, dim=-1)

    # # Compute similarity matrix: [batch_size, num_patches, num_patches]
    # sim_matrix = torch.matmul(student_norm, teacher_norm.transpose(-1, -2)) / temperature

    # # Cross-entropy loss: each student patch predicts corresponding teacher patch
    # labels = torch.arange(num_patches, device=sim_matrix.device).expand(batch_size, -1)
    # loss = F.cross_entropy(
    #     sim_matrix.reshape(-1, num_patches),
    #     labels.reshape(-1),
    #     reduction='mean'
    # )

    # ===== New implementation: only compute loss on masked patches =====
    batch_size, num_patches, emb_dim = student_patches.shape

    # Normalize features
    student_norm = F.normalize(student_patches, dim=-1)
    teacher_norm = F.normalize(teacher_patches, dim=-1)

    # Compute similarity matrix: [batch_size, num_patches, num_patches]
    sim_matrix = torch.matmul(student_norm, teacher_norm.transpose(-1, -2)) / temperature

    # Only compute loss on masked patches
    # mask_patches: [batch_size, num_patches] where 0 indicates masked patches
    masked_indices = (mask_patches == 0).nonzero(as_tuple=False)  # [num_masked, 2] (batch_idx, patch_idx)

    if masked_indices.shape[0] == 0:
        # No masked patches, return zero loss
        return torch.tensor(0.0, device=student_patches.device, requires_grad=True)

    # Extract logits and labels for masked patches only
    batch_indices = masked_indices[:, 0]
    patch_indices = masked_indices[:, 1]

    # Get logits for masked patches: [num_masked, num_patches]
    masked_logits = sim_matrix[batch_indices, patch_indices, :]

    # Labels are the patch indices themselves (self-alignment)
    masked_labels = patch_indices

    # Compute cross-entropy loss only on masked patches
    loss = F.cross_entropy(
        masked_logits,
        masked_labels,
        reduction='mean'
    )

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
    for global_view, local_view in dataloader:
        global_view = global_view.to(device)
        local_view = local_view.to(device)

        # Apply random mask to global view for iBOT
        masked_global_view, mask = apply_mask(global_view, mask_ratio=0.5)

        # Convert mask to patch-level boolean mask
        # mask: [batch_size, channels, height, width]
        # We need: [batch_size, num_patches] where 0 = masked, 1 = visible
        batch_size, channels, height, width = mask.shape
        patch_size = 16
        h_patches = height // patch_size
        w_patches = width // patch_size
        
        # Downsample mask to patch level by taking mean over each patch
        mask_patches = F.avg_pool2d(mask[:, 0:1, :, :], kernel_size=patch_size, stride=patch_size)
        mask_patches = mask_patches.squeeze(1)  # [batch_size, h_patches, w_patches]
        mask_patches = mask_patches.reshape(batch_size, -1)  # [batch_size, num_patches]

        optimizer.zero_grad()

        # Teacher forward pass (using full global view, no mask)
        with torch.no_grad():
            teacher_dino, teacher_mim, teacher_patches = teacher(global_view)
            
            # Update teacher center with EMA
            # batch_center = torch.mean(teacher_dino, dim=0, keepdim=True)
            # teacher_center = center_momentum * teacher_center + (1 - center_momentum) * batch_center

        # Student DINO forward pass (using full local view for CLS token)
        student_dino, _, _ = student(local_view)

        # Student iBOT forward pass (using masked global view for patch tokens)
        _, student_mim, _ = student(masked_global_view)

        # Compute DINO loss (CLS token alignment between student and teacher)
        loss_dino = dino_loss(student_dino, teacher_dino, temperature)

        # Compute iBOT loss (patch token alignment between student and teacher)
        # Only compute on masked patches
        loss_ibot = ibot_loss(student_mim, teacher_mim, mask_patches, temperature)

        # Combined loss
        loss = loss_dino + loss_ibot

        loss.backward()
        optimizer.step()

        # Update teacher EMA
        update_teacher_ema(student, teacher, ema_decay)

        total_loss += loss.item()
        total_dino_loss += loss_dino.item()
        total_ibot_loss += loss_ibot.item()
        step_count += 1

    avg_loss = total_loss / step_count
    avg_dino_loss = total_dino_loss / step_count
    avg_ibot_loss = total_ibot_loss / step_count

    return avg_loss, avg_dino_loss, avg_ibot_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    augmentation = MIMAugmentation(scale=(0.1, 0.5))

    # Dataset
    train_dirs = ['../PCA/train_pic/Sim_1T_256', '../PCA/train_pic/Sim_2H_256']
    dataset = STEMDataset(train_dirs, transform=transform, augmentation=augmentation)
    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4,
                            worker_init_fn=seed_worker,
                            generator=g
                            )

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

    # Initialize teacher center for DINO loss
    # teacher_center = torch.zeros(1, args.dino_out_dim, device=device)

    # Training loop
    for epoch in range(args.epochs):
        loss, dino_loss, ibot_loss = train_epoch(
            student, teacher, dataloader, optimizer, device,
            args.ema_decay, args.temperature
        )
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}, DINO: {dino_loss:.4f}, iBOT: {ibot_loss:.4f}")

        if (epoch + 1) % args.save_freq == 0:
            # Save student weights
            student.encoder.save_pretrained(f"student_weights/student_encoder_epoch_{epoch+1}")
            torch.save(dino_head.state_dict(), f"student_weights/dino_head_epoch_{epoch+1}.pth")
            torch.save(mim_head.state_dict(), f"student_weights/mim_head_epoch_{epoch+1}.pth")

    # Final save
    student.encoder.save_pretrained("student_weights/student_encoder_final")
    torch.save(dino_head.state_dict(), "student_weights/dino_head_final.pth")
    torch.save(mim_head.state_dict(), "student_weights/mim_head_final.pth")

    logging.info("Training complete!")


if __name__ == "__main__":

    set_seed(42)

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
    parser.add_argument("--dino_out_dim", type=int, default=384)  # Much smaller output dimension
    parser.add_argument("--save_freq", type=int, default=10)

    args = parser.parse_args()
    main(args)
