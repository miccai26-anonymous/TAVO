import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    """
    Generic upsampling decoder:
    Takes the last-layer features from EfficientViT backbone (typically 1/32 scale)
    and upsamples back to the original resolution.
    """

    def __init__(self, in_channels, out_channels=128, upsample_scale=32):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Upsample to original resolution
        self.upsample_scale = upsample_scale

    def forward(self, x):
        # x: (B, in_channels, H/32, W/32)
        if isinstance(x, dict):  # handle dict input
            x = list(x.values())[-1]

        x = self.conv_block(x)
        x = F.interpolate(x, scale_factor=self.upsample_scale, mode='bilinear', align_corners=False)
        return x
