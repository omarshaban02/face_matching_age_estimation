import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import cfg

def parse_age_from_filename(fname: str) -> int:
    # UTKFace filenames often format: "<age>_<gender>_<race>_*.jpg"
    base = os.path.basename(fname)
    try:
        age = int(base.split("_")[0].split("/")[-1])
        age = max(0, min(cfg.MAX_AGE, age))
        return age
    except:
        # fallback: if dataset provides explicit field, that will be used instead
        return None

# Dataset wrapper
class UTKFaceHFDataset(Dataset):
    def __init__(self, hf_dataset, transforms=None):
        self.ds = hf_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        # 1) Load image (already decoded by HuggingFace)
        img = item["jpg.chip.jpg"]

        # FORCE RGB
        if img.mode != "RGB":
            img = img.convert("RGB")


        # 2) Parse age from __key__
        key = item["__key__"]         # 'UTKFace/82_0_2_20170111210110290'
        age_str = key.split("/")[1].split("_")[0]
        age = int(age_str)

        # clamp to range
        age = max(0, min(cfg.MAX_AGE, age))

        # 3) Apply transforms (albumentations)
        if self.transforms:
            img_np = np.asarray(img)
            aug = self.transforms(image=img_np)
            img_t = aug["image"]
        else:
            img_t = transforms.ToTensor()(img)

        # 4) CORN binary targets
        K = cfg.MAX_AGE + 1
        bins = torch.zeros(K - 1, dtype=torch.float32)
        for k in range(K - 1):
            bins[k] = 1.0 if age > k else 0.0

        return img_t, bins, age
    
def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    bins = torch.stack([b[1] for b in batch])
    ages = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return imgs, bins, ages

def make_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.ShiftScaleRotate(rotate_limit=15, p=0.4),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            A.Normalize(),
            ToTensorV2()
        ])
