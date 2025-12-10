import torch
import numpy as np
from torch import nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
from model import AgeEstimatorModel
from config import cfg

def load_model_from_hf(repo_id: str, filename, backbone_name: str = "tf_efficientnetv2_s.in21k", max_age: int = 100, device="cuda"):
    """
    Load model from Hugging Face repo.
    """
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = AgeEstimatorModel(backbone_name=backbone_name, pretrained=True, out_dim=max_age+1)
    state_dict = torch.load(local_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def corn_inference(model: nn.Module, pil_img: Image.Image, device="cuda"):
    """
    Run CORN model inference on a single PIL image.
    Returns predicted age (float)
    """
    # Ensure RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Albumentations transform (resize + normalize)
    transform = A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
    img_np = np.array(pil_img)
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    tensor_img = transform(image=img_np)["image"].unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits)
        age_pred = probs.sum(dim=1).item()

    return age_pred