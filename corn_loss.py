from torch import nn
import torch
import torch.nn.functional as F

# CORN Head & Loss
class CORNHead(nn.Module):
    """
    CORN head: outputs K-1 logits; during inference we convert to predicted age by
    summing the probabilities (or thresholds).
    """
    def __init__(self, in_features: int, K: int):
        super().__init__()
        self.K = K  # number of discrete ages (0..MAX_AGE)
        self.linear = nn.Linear(in_features, K - 1)

    def forward(self, x):
        # x: [B, C]
        logits = self.linear(x)  # [B, K-1]
        return logits

def corn_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, K-1] raw scores
    labels: [B, K-1] binary targets where labels[b,k]=1 if age > k else 0
    Use binary cross entropy on each threshold (cumulative link).
    """
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    return bce

def corn_pred_to_age(probs: torch.Tensor) -> torch.Tensor:
    """
    probs: sigmoid(logits) shape [B, K-1]
    We can compute expected age by summing probabilities (CORN approach).
    age = sum_k p_k
    """
    # sum over thresholds
    return probs.sum(dim=1)