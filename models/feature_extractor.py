import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    Feature extractor for EfficientViT_Seg
    ✅ Works perfectly with your current EfficientViT_Seg definition.
    It returns the backbone feature maps (before decoder).
    """

    def __init__(self, seg_model):
        super().__init__()
        self.seg_model = seg_model

    def forward(self, x):
        with torch.no_grad():
            # Extract EfficientViT backbone features
            feats = self.seg_model.model.backbone(x)

            # Handle dict output (EfficientViT may return multi-level features)
            if isinstance(feats, dict):
                feats = list(feats.values())[-1]

        return feats
