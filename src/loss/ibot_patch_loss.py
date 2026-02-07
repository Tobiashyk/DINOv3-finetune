import torch
import torch.nn as nn
import torch.nn.functional as F


def lossfunc(t, s, temp):  # noqa: F811
    return torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)


class SinkhornKnoppTeacher(nn.Module):
    """
    NOTE: This is a module and not a function in the `iBOTPatchLoss` class
    This is because we want to torch.compile it, and torch.compil-ing a single
    function with the `@torch.compile` decorator is bad.
    It's better to `module.compile()` it, as we can control when we enable or
    disable compilation globally.
    """

    @torch.no_grad()
    def forward(
        self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3
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


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.sinkhorn_knopp_teacher = SinkhornKnoppTeacher()
        self.sinkhorn_knopp_teacher.compile()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = lossfunc(t, s, self.student_temp)
        loss = torch.sum(
            loss * student_masks_flat.float(), dim=-1
        ) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        # loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]
