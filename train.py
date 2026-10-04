"""Train LaneUNet from scratch (48x48) with TensorBoard logging."""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import LaneDataset
from model import LaneUNet

DEFAULT_DATA = "/Users/pittachiox/Downloads/data/1000/dataset"


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


def batch_iou(logits, y):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * y).sum((1, 2, 3))
    union = ((pred + y) > 0).float().sum((1, 2, 3))
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()


def run_epoch(model, loader, bce, device, opt=None):
    model.train(opt is not None)
    tot_loss = tot_iou = n = 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = 0.5 * bce(out, y) + 0.5 * dice_loss(out, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_loss += loss.item() * len(x)
            tot_iou += batch_iou(out, y) * len(x)
            n += len(x)
    return tot_loss / n, tot_iou / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/best.pt")
    a = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    torch.manual_seed(0)
    tr = DataLoader(LaneDataset(f"{a.data_dir}/images/train", train=True), a.batch_size, shuffle=True)
    va = DataLoader(LaneDataset(f"{a.data_dir}/images/val"), a.batch_size)
    model = LaneUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    bce = nn.BCEWithLogitsLoss()
    writer = SummaryWriter("runs/lane_unet")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    best = 1e9
    for ep in range(1, a.epochs + 1):
        tl, ti = run_epoch(model, tr, bce, device, opt)
        vl, vi = run_epoch(model, va, bce, device)
        writer.add_scalars("loss", {"train": tl, "val": vl}, ep)
        writer.add_scalars("iou", {"train": ti, "val": vi}, ep)
        print(f"epoch {ep:02d} train_loss {tl:.4f} val_loss {vl:.4f} train_iou {ti:.4f} val_iou {vi:.4f}")
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": vl, "val_iou": vi}, a.out)
    writer.close()
    print("best val_loss", best)


if __name__ == "__main__":
    main()
