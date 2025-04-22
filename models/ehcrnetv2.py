import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from .attention import CBAM
from .aspp import ASPP

class EHCRNetV2(nn.Module):
    def __init__(self, num_classes=46):
        super(EHCRNetV2, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.att = CBAM(1280)  # Should match EfficientNet output channels
        self.aspp = ASPP(1280, [3, 6, 9])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.Mish(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.att(x)
        x = self.aspp(x)
        x = self.classifier(x)
        return x
