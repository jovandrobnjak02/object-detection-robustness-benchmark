import argparse
import csv
import time
from pathlib import Path

import torch
import torch.optim as optim

from datasets.custom_cnn_dataset import get_dataloader
from models.custom_cnn import CustomCNN, CustomCNNLoss

EPOCHS      = 135
BATCH_SIZE  = 4
NUM_WORKERS = 4
IMG_SIZE    = 448
LR          = 1e-3
CHECKPOINTS = Path(__file__).parent.parent / "checkpoints"
LOGS        = Path(__file__).parent.parent / "results" / "logs"


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def run_val(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds  = model(images)
        total_loss += criterion(preds, labels).item()
    return total_loss / len(loader)


def train(use_batchnorm: bool, use_residual: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tag = "baseline"
    if use_batchnorm and use_residual:
        tag = "full"
    elif use_batchnorm:
        tag = "mod_a_batchnorm"
    elif use_residual:
        tag = "mod_b_residual"
    print(f"Run: {tag}")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{tag}.csv"

    train_loader = get_dataloader("train", img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader   = get_dataloader("val",   img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    model     = CustomCNN(use_batchnorm=use_batchnorm, use_residual=use_residual).to(device)
    criterion = CustomCNNLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    start_epoch = 1
    best_val    = float("inf")
    resume_path = CHECKPOINTS / f"{tag}_latest.pt"

    try:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["val_loss"]
        print(f"Resumed from epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")
    except FileNotFoundError:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "epoch_time_s"])

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = run_val(model, val_loader, criterion, device)
        elapsed    = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.0f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.2e}", f"{elapsed:.1f}"])

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "use_batchnorm": use_batchnorm,
            "use_residual": use_residual,
        }
        torch.save(ckpt, CHECKPOINTS / f"{tag}_latest.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, CHECKPOINTS / f"{tag}_best.pt")
            print(f"  -> best val loss: {best_val:.4f}")

    print(f"Training complete. Best val loss: {best_val:.4f}")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CustomCNN on KITTI")
    parser.add_argument("--batchnorm", action="store_true", help="Enable batch normalization (mod A)")
    parser.add_argument("--residual",  action="store_true", help="Enable residual connections (mod B)")
    args = parser.parse_args()

    train(use_batchnorm=args.batchnorm, use_residual=args.residual)
