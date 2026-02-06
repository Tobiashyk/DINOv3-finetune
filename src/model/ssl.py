import torch
import torch.nn as nn


class StudentSSLModel(nn.Module):
    def __init__(self, backbone: nn.Module, dino_head: nn.Module, ibot_head: nn.Module):
        super(StudentSSLModel, self).__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x, is_training=True)
        cls_features = features["x_norm_clstoken"]
        patch_features = features["x_norm_patchtokens"]
        
        dino_output = self.dino_head(cls_features)
        ibot_output = self.ibot_head(patch_features.reshape(-1, patch_features.size(-1)))
        return cls_features, dino_output, ibot_output


class TeacherSSLModel(nn.Module):
    def __init__(self, backbone: nn.Module, dino_head: nn.Module, ibot_head: nn.Module):
        super(TeacherSSLModel, self).__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x, is_training=True)
        cls_features = features["x_norm_clstoken"]
        patch_features = features["x_norm_patchtokens"]

        dino_output = self.dino_head(cls_features)
        ibot_output = self.ibot_head(patch_features.reshape(-1, patch_features.size(-1)))
        return cls_features, dino_output, ibot_output
