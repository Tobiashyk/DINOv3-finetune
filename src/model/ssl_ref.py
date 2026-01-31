"""
Mini-DINOv3 Model Components

This module contains the simplified DINO model components:
1. DINOHead: Projection head that maps features to prototype space
2. MiniDINO: Complete self-distillation model with student/teacher

Based on the original DINOv3 implementation.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .loss import DINOLoss


class DINOHead(nn.Module):
    """
    Projection head for DINO.

    Maps backbone features to a prototype space using a 3-layer MLP.
    The output is L2-normalized before the final projection to prototypes.

    Architecture:
        in_dim -> hidden_dim -> bottleneck_dim -> out_dim

    Args:
        in_dim: Input feature dimension (e.g., 384 for ViT-Small)
        out_dim: Number of prototypes (e.g., 8192)
        hidden_dim: Hidden layer dimension
        bottleneck_dim: Bottleneck dimension before final projection
    """

    def __init__(
        self,
        in_dim: int = 384,
        out_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()

        # Build 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        # Final projection layer (no bias)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection head.

        Args:
            x: Input features [B, D]

        Returns:
            Prototype logits [B, K]
        """
        # Pass through MLP
        x = self.mlp(x)

        # L2 normalize before final projection
        x = F.normalize(x, dim=-1, p=2)

        # Final projection to prototypes
        x = self.last_layer(x)

        return x


class MiniDINO(nn.Module):
    """
    Ultra-minimal DINO self-distillation model.

    This is a simplified educational implementation that demonstrates the core
    concepts of DINO without the complexity of multi-crop, distributed training,
    or additional losses.

    Key components:
    - Student network: trained with gradients
    - Teacher network: updated via EMA (no gradients)
    - DINO loss: cross-entropy with Sinkhorn-Knopp centering

    Args:
        backbone_name: Name of pretrained DINOv3 backbone from torch.hub
        out_dim: Number of prototypes
        hidden_dim: Hidden dimension in projection head
        bottleneck_dim: Bottleneck dimension in projection head
        student_temp: Temperature for student softmax
        teacher_temp: Temperature for teacher softmax
        center_momentum: Momentum for center update
    """

    def __init__(
        self,
        backbone_name: str = 'dinov3_vits14',
        out_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()

        self.teacher_temp = teacher_temp

        # Load pretrained DINOv3 backbone from torch.hub
        print(f"Loading pretrained backbone: {backbone_name}")
        self.student_backbone = torch.hub.load('facebookresearch/dinov3', backbone_name)

        # Get feature dimension from backbone
        # For ViT-Small: 384, ViT-Base: 768, ViT-Large: 1024
        if 'vits' in backbone_name:
            in_dim = 384
        elif 'vitb' in backbone_name:
            in_dim = 768
        elif 'vitl' in backbone_name:
            in_dim = 1024
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Create projection head
        self.student_head = DINOHead(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

        # Teacher is a copy of student (no gradients)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = copy.deepcopy(self.student_head)

        # Freeze teacher
        self.teacher_backbone.requires_grad_(False)
        self.teacher_head.requires_grad_(False)

        # DINO loss
        self.dino_loss = DINOLoss(
            out_dim=out_dim,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )

        print(f"MiniDINO initialized:")
        print(f"  - Backbone: {backbone_name} ({in_dim}D)")
        print(f"  - Prototypes: {out_dim}")
        print(f"  - Student temp: {student_temp}")
        print(f"  - Teacher temp: {teacher_temp}")

    def forward(self, images: torch.Tensor) -> dict:
        """
        Forward pass: compute loss between student and teacher.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            Dictionary containing:
                - loss: DINO loss value
                - student_logits: Student output logits
                - teacher_probs: Teacher output probabilities
        """
        # Student forward pass
        student_features = self.student_backbone(images)['x_norm_clstoken']  # [B, D]
        student_logits = self.student_head(student_features)  # [B, K]

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_features = self.teacher_backbone(images)['x_norm_clstoken']  # [B, D]
            teacher_logits = self.teacher_head(teacher_features)  # [B, K]

            # Center teacher outputs
            teacher_logits_centered = teacher_logits - self.dino_loss.center

            # Apply Sinkhorn-Knopp to get teacher probabilities
            teacher_probs = self.dino_loss.sinkhorn_knopp_teacher(
                teacher_logits_centered,
                teacher_temp=self.teacher_temp,
            )

            # Update center
            self.dino_loss.update_center(teacher_logits)

        # Compute DINO loss
        loss = self.dino_loss(student_logits, teacher_probs)

        return {
            'loss': loss,
            'student_logits': student_logits,
            'teacher_probs': teacher_probs,
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996):
        """
        Update teacher with exponential moving average of student.

        Teacher parameters are updated as:
            teacher = momentum * teacher + (1 - momentum) * student

        Args:
            momentum: EMA momentum (typically 0.996)
        """
        # Update backbone
        for param_s, param_t in zip(
            self.student_backbone.parameters(),
            self.teacher_backbone.parameters()
        ):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)

        # Update head
        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)

    def get_student_parameters(self):
        """Get parameters that should be optimized (student only)."""
        return list(self.student_backbone.parameters()) + list(self.student_head.parameters())
