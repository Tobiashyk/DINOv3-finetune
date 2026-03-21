import torch
import torch.nn.functional as F


def noise_loss_calculate(student_patches, teacher_patches, temperature=0.1):
    """
    第二阶段Noise Loss: 纯粹的去噪特征蒸馏损失
    计算Student(加噪图)和Teacher(干净图)的Patch Token之间的交叉熵损失

    Args:
        student_patches: Student模型输出的patch tokens [B, N, D]
        teacher_patches: Teacher模型输出的patch tokens [B, N, D]
        temperature: 温度参数，默认0.1

    Returns:
        loss: 标量损失值
    """
    batch_size = student_patches.size(0)
    num_patches = student_patches.size(1)
    total_loss = 0.0

    for b in range(batch_size):
        # L2归一化
        student_norm = F.normalize(student_patches[b], dim=-1)  # [num_patches, emb_dim]
        teacher_norm = F.normalize(teacher_patches[b], dim=-1)  # [num_patches, emb_dim]

        # 计算相似度矩阵
        sim_matrix = torch.matmul(student_norm, teacher_norm.T) / temperature  # [N, N]

        # 对角线为正样本(对应位置的patch)
        labels = torch.arange(num_patches, device=sim_matrix.device)

        # 交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        total_loss += loss

    return total_loss / batch_size
