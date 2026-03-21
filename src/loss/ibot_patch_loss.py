import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornKnopp(nn.Module):
    """
    Single-GPU Sinkhorn-Knopp normalizer for teacher soft labels.

    Transforms teacher logits into a doubly stochastic assignment matrix,
    ensuring each prototype is used uniformly across patches (prevents collapse).

    Reference: DINOv3 dinov3/loss/ibot_patch_loss.py::SinkhornKnoppTeacher
    """

    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        """
        Args:
            teacher_output:           [total_masked, K]  raw logits from teacher ibot_head
            teacher_temp:             float              temperature
            n_masked_patches_tensor:  scalar tensor      total number of masked patches
            n_iterations:             int                number of SK normalization rounds

        Returns:
            soft_labels: [total_masked, K]  probability assignment (columns sum to 1)
        """
        teacher_output = teacher_output.float()
        Q = torch.exp((teacher_output - teacher_output.max(dim=-1, keepdim=True).values) / teacher_temp).t()  # [K, total_masked]
        B = n_masked_patches_tensor  # total masked patches (scalar tensor)
        K = Q.shape[0]              # number of prototypes

        # Normalize matrix to sum to 1
        Q = Q / Q.sum()

        for _ in range(n_iterations):
            # Row normalization: each prototype contributes equally (weight 1/K)
            Q = Q / Q.sum(dim=1, keepdim=True)
            Q = Q / K
            # Column normalization: each patch assigns equal total weight (1/B)
            Q = Q / Q.sum(dim=0, keepdim=True)
            Q = Q / B

        Q = Q * B  # columns now sum to 1 (valid assignment distributions)
        return Q.t()  # [total_masked, K]


class iBOTPatchLoss(nn.Module):
    """
    iBOT patch-level self-supervised loss.

    Replaces the old InfoNCE formulation with the DINOv3 paradigm:
      - Token-level masking (applied in backbone via mask_token)
      - Sinkhorn-Knopp soft labels from teacher
      - Cross-entropy between student logits and teacher SK soft labels
      - EMA center buffer for optional softmax-center alternative

    Reference: DINOv3 dinov3/loss/ibot_patch_loss.py::iBOTPatchLoss
    """

    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        # Center buffer shape [1, 1, K] so it broadcasts over [B, N, K]
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.sinkhorn_knopp = SinkhornKnopp()

    def sinkhorn_knopp_teacher(self, teacher_masked_logits, teacher_temp, n_masked_patches_tensor):
        """Apply SK normalization to flattened masked teacher logits.

        Args:
            teacher_masked_logits:   [total_masked, K]  raw logits
            teacher_temp:            float
            n_masked_patches_tensor: scalar tensor       total masked count

        Returns:
            soft_labels: [total_masked, K]
        """
        return self.sinkhorn_knopp(teacher_masked_logits, teacher_temp, n_masked_patches_tensor)

    def forward_masked(
        self,
        student_masked_logits,
        teacher_soft_labels,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        """Cross-entropy loss on masked patches only.

        Args:
            student_masked_logits: [total_masked, K]  student raw logits (pre-softmax)
            teacher_soft_labels:   [total_masked, K]  SK-normalized teacher probabilities
            student_masks_flat:    [B, N]             bool mask (True = masked)
            n_masked_patches:      optional int        limit to first n patches
            masks_weight:          optional [total_masked]  per-patch weights

        Returns:
            scalar loss
        """
        # KL: -sum(teacher * log_softmax(student / temp))  =>  [total_masked]
        loss = torch.sum(
            teacher_soft_labels.float() * F.log_softmax(
                student_masked_logits.float() / self.student_temp, dim=-1
            ),
            dim=-1,
        )

        if masks_weight is None:
            # Weight each patch by 1 / (# masked patches in its sample)
            # shape arithmetic: [B] -> [B,1] -> [B,N] -> [total_masked]
            masks_weight = (
                (1.0 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )

        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]

        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_logits, teacher_temp):
        """Alternative to SK: center subtraction + softmax (for reference / debugging).

        Applies the pending EMA center update before computing probabilities.

        Args:
            teacher_patch_logits: [..., K]
            teacher_temp:         float

        Returns:
            soft_labels: [..., K]
        """
        return F.softmax((teacher_patch_logits - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_patch_logits):
        """Synchronous EMA update of center buffer.

        Args:
            teacher_patch_logits: [B, N, K]  all teacher patch logits (after ibot_head)
        """
        # Mean over patches per sample, then mean over batch  =>  [1, 1, K]
        batch_center = (
            teacher_patch_logits.mean(dim=1)  # [B, K]
            .mean(dim=0)                       # [K]
            .unsqueeze(0).unsqueeze(0)         # [1, 1, K]
        )
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
