import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.SiLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply channel attention
        channel_att = self.channel_att(x)
        x_channel = x * channel_att
        
        # Apply spatial attention
        spatial_avg = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_max = torch.max(x_channel, dim=1, keepdim=True)[0]
        spatial_att = self.spatial_att(torch.cat([spatial_avg, spatial_max], dim=1))
        
        return x_channel * spatial_att
