"""
Mini-DINOv3 Training Script

This is a simplified training script that demonstrates how to train the MiniDINO model.
It includes basic data loading, augmentation, and a training loop.

Usage:
    python src/model/train_mini_dino.py --data_dir /path/to/images --epochs 10
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from .ssl_ref import MiniDINO


class SimpleImageDataset(Dataset):
    """
    Simple dataset that loads images from a directory.

    Args:
        data_dir: Directory containing images
        transform: Torchvision transforms to apply
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Find all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))

        print(f"Found {len(self.image_paths)} images in {data_dir}")

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def get_augmentation_transform(image_size: int = 224):
    """
    Create data augmentation transform for DINO training.

    This is a simplified version with single global crop.

    Args:
        image_size: Size of output images

    Returns:
        Torchvision transform
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_mini_dino(
    data_dir: str,
    output_dir: str = 'outputs/mini_dino',
    backbone_name: str = 'dinov3_vits14',
    out_dim: int = 8192,
    num_epochs: int = 10,
    batch_size: int = 8,
    lr: float = 0.001,
    weight_decay: float = 0.04,
    teacher_momentum: float = 0.996,
    device: str = 'cuda',
):
    """
    Train MiniDINO model.

    Args:
        data_dir: Directory containing training images
        output_dir: Directory to save checkpoints and logs
        backbone_name: Name of pretrained backbone
        out_dim: Number of prototypes
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        teacher_momentum: EMA momentum for teacher update
        device: Device to train on ('cuda' or 'cpu')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("\n" + "="*50)
    print("Initializing MiniDINO model...")
    print("="*50)
    model = MiniDINO(
        backbone_name=backbone_name,
        out_dim=out_dim,
    ).to(device)

    # Setup optimizer (only student parameters)
    optimizer = torch.optim.AdamW(
        model.get_student_parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Setup data
    print("\n" + "="*50)
    print("Loading dataset...")
    print("="*50)
    transform = get_augmentation_transform(image_size=224)
    dataset = SimpleImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = outputs['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.get_student_parameters(), max_norm=3.0)

            optimizer.step()

            # Update teacher with EMA
            model.update_teacher(momentum=teacher_momentum)

            # Logging
            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
            })

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Train Mini-DINOv3')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, default='outputs/mini_dino', help='Output directory')
    parser.add_argument('--backbone', type=str, default='dinov3_vits14',
                        choices=['dinov3_vits14', 'dinov3_vitb14', 'dinov3_vitl14'],
                        help='Backbone architecture')
    parser.add_argument('--out_dim', type=int, default=8192, help='Number of prototypes')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.04, help='Weight decay')
    parser.add_argument('--teacher_momentum', type=float, default=0.996, help='Teacher EMA momentum')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')

    args = parser.parse_args()

    train_mini_dino(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backbone_name=args.backbone,
        out_dim=args.out_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        teacher_momentum=args.teacher_momentum,
        device=args.device,
    )


if __name__ == '__main__':
    main()
