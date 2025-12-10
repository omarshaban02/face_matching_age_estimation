import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_prepare import UTKFaceHFDataset, make_transforms, collate_fn
from model import AgeEstimatorModel
from config import cfg
from corn_loss import corn_loss, corn_pred_to_age


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    cnt = 0
    pbar = tqdm(enumerate(loader), total=len(loader))
    for i, (imgs, bins, ages) in pbar:
        imgs = imgs.to(device)
        bins = bins.to(device)
        ages = ages.to(device)

        logits = model(imgs)  # [B, K-1]
        loss = corn_loss(logits, bins)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = corn_pred_to_age(probs)  # expected age
            mae = torch.abs(preds - ages).mean().item()

        running_loss += loss.item() * imgs.size(0)
        running_mae += mae * imgs.size(0)
        cnt += imgs.size(0)

        if i % cfg.PRINT_FREQ == 0:
            pbar.set_description(f"Epoch {epoch} loss={running_loss/cnt:.4f} mae={running_mae/cnt:.3f}")

    return running_loss / cnt, running_mae / cnt

def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    cnt = 0
    with torch.no_grad():
        for imgs, bins, ages in tqdm(loader, desc="Val"):
            imgs = imgs.to(device)
            bins = bins.to(device)
            ages = ages.to(device)

            logits = model(imgs)
            loss = corn_loss(logits, bins)

            probs = torch.sigmoid(logits)
            preds = corn_pred_to_age(probs)
            mae = torch.abs(preds - ages).mean().item()

            running_loss += loss.item() * imgs.size(0)
            running_mae += mae * imgs.size(0)
            cnt += imgs.size(0)

    return running_loss / cnt, running_mae / cnt