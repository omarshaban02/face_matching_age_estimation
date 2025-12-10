from torch import nn
import timm
from corn_loss import CORNHead

# Model build
class AgeEstimatorModel(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool, out_dim: int):
        super().__init__()
        # load timm backbone
        model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = model.num_features
        self.backbone = model
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            CORNHead(512, out_dim)
        )

    def forward(self, x):
        f = self.backbone(x)  # [B, feat_dim]
        logits = self.head(f) # [B, K-1]
        return logits