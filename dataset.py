"""Dataset utilities: YOLO-seg polygon -> binary mask, resizing, augmentation."""
import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 48
LANE_CLASS = 0  # only class 0 (lane) polygons are used


def read_polygons(label_path, cls=LANE_CLASS):
    """Return list of (N,2) normalized polygons of class `cls`. Rows with <3 points (bbox/other) are skipped."""
    polys = []
    if not os.path.exists(label_path):
        return polys
    for line in open(label_path):
        p = line.split()
        if len(p) < 7 or int(float(p[0])) != cls:
            continue
        polys.append(np.array(p[1:], dtype=np.float32).reshape(-1, 2))
    return polys


def polygons_to_mask(polys, h, w):
    """Rasterize normalized polygons at size (h, w) -> uint8 {0,1}."""
    mask = np.zeros((h, w), np.uint8)
    for poly in polys:
        pts = np.round(poly * [w, h]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def list_images(img_dir):
    files = []
    for ext in ("jpg", "jpeg", "png"):
        files += glob.glob(os.path.join(img_dir, f"*.{ext}"))
    return sorted(files)


def label_path_for(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    return os.path.join(os.path.dirname(img_path).replace("images", "labels"), base)


def preprocess(img_bgr, size=IMG_SIZE):
    """BGR uint8 (any size) -> float tensor (3,size,size) RGB in [0,1]."""
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def augment(img):
    """Slight photometric augmentation on RGB float image (H,W,3) in [0,1]: white-balance, brightness, blur."""
    if random.random() < 0.5:  # white balance shift: per-channel gain
        img = img * np.random.uniform(0.92, 1.08, size=(1, 1, 3))
    if random.random() < 0.5:  # brightness shift
        img = img + random.uniform(-0.08, 0.08)
    if random.random() < 0.3:  # small blur
        img = cv2.GaussianBlur(img.astype(np.float32), (3, 3), random.uniform(0.3, 0.8))
    return np.clip(img, 0, 1).astype(np.float32)


class LaneDataset(Dataset):
    def __init__(self, img_dir, train=False, size=IMG_SIZE):
        self.files, self.train, self.size = list_images(img_dir), train, size
        if not self.files:
            raise FileNotFoundError(f"no images in {img_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        bgr = cv2.imread(path)
        # rasterize at the image's own resolution, then downsize (various image sizes supported)
        mask = polygons_to_mask(read_polygons(label_path_for(path)), bgr.shape[0], bgr.shape[1])
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = preprocess(bgr, self.size)
        if self.train:
            x = torch.from_numpy(augment(x.permute(1, 2, 0).numpy())).permute(2, 0, 1)
        return x, torch.from_numpy(mask.astype(np.float32))[None]
