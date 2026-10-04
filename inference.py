"""Run lane detection; save binary masks (0/255 PNG, original image size) under run/."""
import argparse
import os

import cv2
import numpy as np
import torch

from dataset import list_images, preprocess
from model import LaneUNet

DEFAULT_DATA = "/Users/pittachiox/Downloads/data/1000/dataset"


def load_model(ckpt, device):
    m = LaneUNet().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return m.eval()


@torch.no_grad()
def predict_mask(model, bgr, device):
    """BGR image -> binary mask (uint8 0/1) at the original image size."""
    x = preprocess(bgr)[None].to(device)
    p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    small = (p > 0.5).astype(np.uint8)
    return cv2.resize(small, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=f"{DEFAULT_DATA}/images/test")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="run")
    a = ap.parse_args()
    device = torch.device("cpu")
    model = load_model(a.ckpt, device)
    os.makedirs(a.out, exist_ok=True)
    files = list_images(a.img_dir)
    for f in files:
        mask = predict_mask(model, cv2.imread(f), device)
        name = os.path.splitext(os.path.basename(f))[0] + ".png"
        cv2.imwrite(os.path.join(a.out, name), mask * 255)
    print(f"saved {len(files)} masks to {a.out}/")


if __name__ == "__main__":
    main()
