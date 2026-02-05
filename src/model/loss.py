"""
DINO Loss Function Module

Implements the DINO (Self-Distillation with No Labels) self-supervised learning loss function.
This loss function uses knowledge distillation to make the student network learn the output distribution of the teacher network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    DINO Loss Function Class

    Uses cross-entropy loss to measure the difference between student and teacher network outputs.
    Teacher network outputs are normalized using the Sinkhorn-Knopp algorithm to avoid mode collapse.
    """

    def __init__(self, out_dim: int, student_temp: float = 0.1, center_momentum: float = 0.9):
        """
        Initialize DINO loss function

        Args:
            out_dim: Output dimension (number of prototypes)
            student_temp: Temperature parameter for student network, used for softmax normalization, lower temperature makes distribution sharper
            center_momentum: Momentum parameter for center update, used for exponential moving average
        """
        super().__init__()
        self.student_temp = student_temp  # Student temperature parameter
        self.center_momentum = center_momentum  # Center momentum parameter

        # Register a buffer to store the center of teacher outputs for centering operation
        self.register_buffer("center", torch.zeros(1, out_dim))

    def sinkhorn_knopp_teacher(self, teacher_output: torch.Tensor, teacher_temp: float, n_iterations: int = 3) -> torch.Tensor:
        """
        Normalize teacher network output using Sinkhorn-Knopp algorithm

        The Sinkhorn-Knopp algorithm converts teacher output into a doubly stochastic matrix through iterative row-column normalization,
        which helps prevent mode collapse and ensures all prototypes are used uniformly.

        Args:
            teacher_output: Teacher network output logits, shape [B, K]
            teacher_temp: Temperature parameter for teacher network
            n_iterations: Number of Sinkhorn-Knopp iterations

        Returns:
            Normalized teacher probability distribution, shape [B, K]
        """
        teacher_output = teacher_output.float()

        # Use log-sum-exp trick for numerical stability, subtract max to prevent exponential overflow
        Q = torch.exp((teacher_output) / teacher_temp).t()  # [K, B]
        B = Q.shape[1]  # Batch size
        K = Q.shape[0]  # Number of prototypes

        # Normalize matrix so its sum equals 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        # Sinkhorn-Knopp iteration process
        for _ in range(n_iterations):
            # Row normalization: total weight for each prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= (sum_of_rows + 1e-8)  # Add epsilon to avoid division by zero
            Q /= K

            # Column normalization: total weight for each sample must be 1/B
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= (sum_of_cols + 1e-8)  # Add epsilon to avoid division by zero
            Q /= B

        # Scale back so column sums equal 1 (valid probability distribution)
        Q *= B
        return Q.t()  # [B, K]

    def forward(self, student_probs: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate DINO loss

        Uses cross-entropy loss to measure the difference between student and teacher network output distributions.
        Student network outputs are converted to probability distributions through temperature-scaled softmax.

        Args:
            student_logits: Student network output logits, shape [B, K]
            teacher_probs: Teacher network probability distribution (already normalized by Sinkhorn-Knopp), shape [B, K]

        Returns:
            Scalar loss value
        """
        S, B, K = student_probs.shape
        T, _, _ = teacher_probs.shape


        # Apply temperature-scaled log_softmax to student network logits
        student_log_probs = F.log_softmax(student_probs / self.student_temp, dim=-1)

        # Calculate cross-entropy loss: -sum(teacher_probs * log(student_probs))
        # loss = -torch.sum(teacher_probs * student_log_probs, dim=-1)
        loss = -torch.einsum("s b k, t b k -> ", student_log_probs, teacher_probs)

        return loss / (B * S * T)

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """
        Update the center of teacher outputs

        Uses exponential moving average (EMA) to update the center vector. The center vector is used for centering operation on teacher outputs,
        which helps avoid one dimension dominating the output, thus preventing mode collapse.

        Args:
            teacher_output: Teacher network output, shape [B, K]
        """
        # Calculate center of current batch
        batch_center = teacher_output.mean(dim=0, keepdim=True)

        # Update global center using exponential moving average
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
