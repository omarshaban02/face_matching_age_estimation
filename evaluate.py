import torch
from torch import nn
from torch.utils.data import DataLoader

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = "cuda"):
    """
    Compute Mean Absolute Error (MAE) on a dataset
    Args:
        model: PyTorch model
        dataloader: DataLoader providing (img, bins, age)
        device: 'cuda' or 'cpu'
    Returns:
        mae: float
    """
    model.eval()
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, bins, ages in dataloader:
            imgs = imgs.to(device)
            ages = ages.to(device)
            bins = bins.to(device)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = probs.sum(dim=1)  # CORN expected age

            total_mae += torch.sum(torch.abs(preds - ages)).item()
            total_samples += imgs.size(0)

    mae = total_mae / total_samples
    print(f"Evaluation MAE: {mae:.3f}")
    return mae
