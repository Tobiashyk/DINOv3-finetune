import torch
import torch.nn as nn


class StudentSSLModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super(StudentSSLModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        projections = self.head(features)
        return projections
    
class TeacherSSLModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super(TeacherSSLModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        projections = self.head(features)
        return projections