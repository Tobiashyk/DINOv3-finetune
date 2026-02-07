import torch
import torch.nn as nn
from typing import List


class StudentSSLModel(nn.Module):
    def __init__(self, backbone: nn.Module, dino_head: nn.Module, ibot_head: nn.Module):
        super(StudentSSLModel, self).__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head

    def forward(self, x: List[torch.Tensor], masks: List[torch.Tensor | None]) -> dict[str, torch.Tensor]:
        features = self.backbone(x, masks=masks, is_training=True)
        
        global_cls_features = features[0]["x_norm_clstoken"]
        global_patch_features = features[0]["x_norm_patchtokens"]
        local_cls_features = features[1]["x_norm_clstoken"]
        local_patch_features = features[1]["x_norm_patchtokens"]

        masks = features[0]["masks"]
        
        global_dino_output = self.dino_head(global_cls_features)
        global_ibot_output = self.ibot_head(global_patch_features.flatten(0, 1))
        local_dino_output = self.dino_head(local_cls_features)
        local_ibot_output = self.ibot_head(local_patch_features.flatten(0, 1))
        return {
            "global_cls_features": global_cls_features,
            "global_dino_output": global_dino_output,
            "global_ibot_output": global_ibot_output,
            "local_cls_features": local_cls_features,
            "local_dino_output": local_dino_output,
            "local_ibot_output": local_ibot_output,
            "masks": masks,
        }


class TeacherSSLModel(nn.Module):
    def __init__(self, backbone: nn.Module, dino_head: nn.Module, ibot_head: nn.Module):
        super(TeacherSSLModel, self).__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head
        self.sinkhorn_knopp_teacher = SinkhornKnoppCentering()
        self.sinkhorn_knopp_teacher.compile()

    @torch.no_grad()
    def forward(self, x: List[torch.Tensor], masks: List[torch.Tensor | None], teacher_temp: float = 0.04) -> dict[str, torch.Tensor]:
        features = self.backbone(x, masks=masks, is_training=True)

        global_cls_features = features[0]["x_norm_clstoken"]
        global_patch_features = features[0]["x_norm_patchtokens"]
        local_cls_features = features[1]["x_norm_clstoken"]
        local_patch_features = features[1]["x_norm_patchtokens"]

        global_dino_output = self.dino_head(global_cls_features)
        global_ibot_output = self.ibot_head(global_patch_features.flatten(0, 1))
        local_dino_output = self.dino_head(local_cls_features)
        local_ibot_output = self.ibot_head(local_patch_features.flatten(0, 1))

        masks = features[0]["masks"]
        n_masked_patches_tensor = masks.sum(dim=-1).sum()

        # Apply Sinkhorn-Knopp centering to teacher outputs
        global_dino_centered = self.sinkhorn_knopp_teacher(global_dino_output, teacher_temp, n_masked_patches_tensor)
        local_dino_centered = self.sinkhorn_knopp_teacher(local_dino_output, teacher_temp, n_masked_patches_tensor)
        global_ibot_centered = self.sinkhorn_knopp_teacher(global_ibot_output, teacher_temp, n_masked_patches_tensor)
        local_ibot_centered = self.sinkhorn_knopp_teacher(local_ibot_output, teacher_temp, n_masked_patches_tensor)

        return {
            "global_cls_features": global_cls_features,
            "global_dino_output": global_dino_output,
            "global_ibot_output": global_ibot_output,
            "local_cls_features": local_cls_features,
            "local_dino_output": local_dino_output,
            "local_ibot_output": local_ibot_output,
            "masks": masks,
            "global_dino_centered": global_dino_centered,
            "local_dino_centered": local_dino_centered,
            "global_ibot_centered": global_ibot_centered,
            "local_ibot_centered": local_ibot_centered,
        }

    @torch.no_grad()
    def update_from_student_ema(self, student: StudentSSLModel, momentum: float = 0.996):
        """
        Update teacher parameters using Exponential Moving Average (EMA) of student parameters.

        Formula: teacher_param = momentum * teacher_param + (1 - momentum) * student_param

        This ensures the teacher network evolves slowly and provides stable targets for the student.

        Args:
            student: Student model (trained with gradients)
            momentum: EMA momentum (default: 0.996)
                      - Higher values (e.g., 0.999) = slower teacher updates
                      - Lower values (e.g., 0.99) = faster teacher updates
                      - Standard DINOv3 setting: 0.996 (effective window ~250 steps)
        """
        for student_param, teacher_param in zip(student.parameters(), self.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)


class SinkhornKnoppCentering(nn.Module):
    """
    NOTE: This is a module and not a function in the `iBOTPatchLoss` class
    This is because we want to torch.compile it, and torch.compil-ing a single
    function with the `@torch.compile` decorator is bad.
    It's better to `module.compile()` it, as we can control when we enable or
    disable compilation globally.
    """

    @torch.no_grad()
    def forward(
        self, teacher_output: torch.Tensor, teacher_temp: float, n_masked_patches_tensor: torch.Tensor, n_iterations: int = 3
    ):
        teacher_output = teacher_output.float()
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = n_masked_patches_tensor
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()