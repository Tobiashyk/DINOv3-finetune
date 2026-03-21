import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class StudentSSLModel(nn.Module):
    def __init__(self, backbone, dino_head, ibot_head):
        super().__init__()
        self.encoder = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head

    def forward(self, x, masks=None):
        """
        Args:
            x:     [B, C, H, W]   input image
            masks: [B, N_patches] bool tensor, True = token will be replaced by mask_token.
                   Pass None for the teacher or for local crops (no masking).

        Returns:
            dino_output: [B, dino_out_dim] or None
            ibot_output: [B, N_patches, ibot_out_dim] or None
            cls_token:   [B, D] raw backbone CLS token (for KoLeo loss)
        """
        features = self.encoder.forward_features(x, masks=masks)
        cls_token = features['x_norm_clstoken']      # [B, D]
        patch_tokens = features['x_norm_patchtokens'] # [B, N, D]

        dino_output = self.dino_head(cls_token) if self.dino_head is not None else None
        ibot_output = self.ibot_head(patch_tokens) if self.ibot_head is not None else None

        return dino_output, ibot_output, cls_token


def generate_token_masks(batch_size, n_patches, mask_ratio, device):
    """Generate random token-level boolean masks at the patch grid resolution.

    Each sample in the batch independently has `floor(n_patches * mask_ratio)` patches
    randomly selected for masking.  The backbone's `prepare_tokens_with_masks` will
    replace those patch embeddings with the learned `mask_token` vector.

    Args:
        batch_size: B
        n_patches:  total patches per image (H_patches * W_patches)
        mask_ratio: fraction of patches to mask  (e.g. 0.5)
        device:     torch.device

    Returns:
        mask: [B, N_patches]  bool tensor, True = masked
    """
    num_masked = int(n_patches * mask_ratio)
    mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)
    for b in range(batch_size):
        masked_indices = torch.randperm(n_patches, device=device)[:num_masked]
        mask[b, masked_indices] = True
    return mask


def apply_lora_to_dino(model, lora_config):
    config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=["qkv"],
        lora_dropout=lora_config['lora_dropout'],
        bias="none",
        modules_to_save=[]
    )
    peft_model = get_peft_model(model, config)

    for name, param in peft_model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    return peft_model
