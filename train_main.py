import torch
from torch.utils.data import DataLoader
from dataset_prepare import UTKFaceHFDataset, make_transforms, collate_fn
from model import AgeEstimatorModel
from config import cfg

def main():
    try:
        ds_info = load_dataset(cfg.HF_DATASET, split="train", streaming=False)
        split_name = "train"
    except Exception:
        # fallback single split
        ds_info = load_dataset(cfg.HF_DATASET, split="train")
        split_name = "train"

    from datasets import load_dataset
    import numpy as np

    full = load_dataset("py97/UTKFace-Cropped", split="train")

    # Keep only rows where 'jpg.chip.jpg' is not None
    valid_indices = [i for i, item in enumerate(full) if item["jpg.chip.jpg"] is not None]
    clean_ds = full.select(valid_indices)

    print(f"Original dataset size: {len(full)}, after cleaning: {len(clean_ds)}")

    # 90/10 split
    n = len(clean_ds)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    cut = int(0.9 * n)
    train_hf = clean_ds.select(idxs[:cut])
    val_hf   = clean_ds.select(idxs[cut:])

    train_dataset = UTKFaceHFDataset(train_hf, transforms=make_transforms(True))
    val_dataset   = UTKFaceHFDataset(val_hf, transforms=make_transforms(False))

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                            shuffle=True, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn)

    # Build model
    model = AgeEstimatorModel(cfg.BACKBONE, cfg.PRETRAINED, cfg.MAX_AGE + 1)
    model.to(cfg.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    best_mae = 1e9

if __name__ == "__main__":
    main()