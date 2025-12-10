import os
import torch

class Config:
    HF_DATASET = "py97/UTKFace-Cropped"   # Hugging Face dataset id
    IMAGE_SIZE = 224
    BACKBONE = "tf_efficientnetv2_s.in21k"  # timm name; some repos provide variants
    PRETRAINED = True
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    LR = 3e-4
    EPOCHS = 30
    MAX_AGE = 116
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTDIR = "outputs_corn"
    PRINT_FREQ = 50
    WEIGHT_DECAY = 1e-4

cfg = Config()
os.makedirs(cfg.OUTDIR, exist_ok=True)