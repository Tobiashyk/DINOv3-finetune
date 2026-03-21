import torch
import torch.nn as nn
import torch.nn.functional as F


def dino_loss_calculate(student_output, teacher_output, center, student_temp=0.1, teacher_temp=0.04, ignore_diagonal=True):
    """
    Multi-crop DINO loss 计算，支持 Teacher-Student 不同数量的 crop

    Args:
        student_output: student 模型输出的 cls token [num_student_crops, batch, out_dim]
                       包含 global crops + local crops
        teacher_output: teacher 模型输出的 cls token [num_teacher_crops, batch, out_dim]
                       只包含 global crops
        center: 教师特征中心点 [1, out_dim]，通过 EMA 更新
        student_temp: student 温度参数 (默认 0.1)
        teacher_temp: teacher 温度参数 (默认 0.04)
        ignore_diagonal: 是否忽略对角线（相同 global view 之间的配对），默认 True

    Returns:
        loss: DINO loss (对所有有效配对求平均)
    """
    # Student logits: [num_student_crops, batch, out_dim]
    student_logits = student_output / student_temp
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # [S, B, K]

    # Teacher logits: [num_teacher_crops, batch, out_dim]
    teacher_logits = (teacher_output - center) / teacher_temp
    teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)  # [T, B, K]

    student_crops, B, K = student_log_probs.shape
    teacher_crops, _, _ = teacher_probs.shape

    if not ignore_diagonal:
        # 计算所有配对：student_crops × teacher_crops
        # loss = -sum(student_log_probs * teacher_probs) / (B * S * T)
        loss = -torch.einsum("s b k, t b k -> ", student_log_probs, teacher_probs)
        return loss / (B * student_crops * teacher_crops)
    else:
        # 忽略对角线（只针对 global crops 部分）
        # 计算所有配对的损失矩阵 [S, T]
        loss_matrix = -torch.einsum("s b k, t b k -> s t", student_log_probs, teacher_probs)  # [S, T]

        # 只对 global crops 部分（前 min(S, T) 个）忽略对角线
        min_global_crops = min(student_crops, teacher_crops)
        for i in range(min_global_crops):
            loss_matrix[i, i] = 0.0

        # 总配对数 = S * T - min(S, T)（因为对角线有 min(S,T) 个被设为0）
        total_pairs = student_crops * teacher_crops - min_global_crops
        return loss_matrix.sum() / (B * total_pairs)


class DINOLoss(nn.Module):
    """
    DINO Loss 模块，支持多 crop 计算

    参考: DINOv3/dinov3/loss/dino_clstoken_loss.py
    """
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_logits, teacher_probs, ignore_diagonal=True):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.

        Args:
            student_logits: [student_crops, batch, prototypes]
            teacher_probs:  [teacher_crops, batch, prototypes] must sum to 1 over the last dim
            ignore_diagonal: 是否忽略对角线配对

        Returns:
            loss: 对所有有效配对求平均的交叉熵损失
        """
        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape

        student_log_probs = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)

        if not ignore_diagonal:
            # 所有配对都计算
            loss = -torch.einsum("s b k, t b k -> ", student_log_probs, teacher_probs)
            return loss / (B * student_crops * teacher_crops)
        else:
            # 计算所有配对的损失矩阵
            loss_matrix = -torch.einsum("s b k, t b k -> s t", student_log_probs, teacher_probs)

            # 对 global crops 部分忽略对角线
            min_global_crops = min(student_crops, teacher_crops)
            for i in range(min_global_crops):
                loss_matrix[i, i] = 0.0

            total_pairs = student_crops * teacher_crops - min_global_crops
            return loss_matrix.sum() / (B * total_pairs)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        更新教师特征中心点

        Args:
            teacher_output: [teacher_crops, batch, out_dim] 或 [batch, out_dim]
        """
        if teacher_output.dim() == 3:
            # Multi-crop: 对所有 teacher crops 求平均
            teacher_output = teacher_output.mean(dim=0)  # [batch, out_dim]

        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)  # [1, out_dim]
        batch_center = batch_center / teacher_output.shape[0]

        # EMA 更新
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
