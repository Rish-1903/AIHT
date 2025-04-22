import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        ))
        
        # Dilated convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, 256, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(256),
                nn.SiLU()
            ))
        
        # Global context
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # Fusion layer
        self.project = nn.Sequential(
            nn.Conv2d(256 * (len(atrous_rates)+2), 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        
        # Upsample the global context feature
        res[-1] = F.interpolate(res[-1], size=x.size()[2:], mode='bilinear', align_corners=False)
        
        # Concatenate all features
        x = torch.cat(res, dim=1)
        return self.project(x)
